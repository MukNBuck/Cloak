"""Local-only runtime settings for air-gapped NDA deployments.

Nothing here talks to the public internet. Ollama and Hugging Face models
must already be on disk; set CLOAK_OFFLINE=1 after the first cache.
"""

from functools import lru_cache
import csv
import json
import os
import sys

# Python 3.14: transformers -> importlib.metadata.packages_distributions()
# parses dist-info RECORD as CSV and dies on fields > 131072.
_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(_limit)
        break
    except OverflowError:
        _limit = min(_limit, 2**31 - 1) // 2
        if _limit < 131072:
            break

from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


def _stub_sklearn_if_scipy_broken():
    """Python 3.14 scipy wheels can ship NUL-corrupted sources.

    transformers still imports sklearn/scipy at module load even for encoder-only
    BGE models. Stub the symbols so embed/rerank can import.
    """
    import importlib.machinery
    import types

    def _broken(modname):
        try:
            __import__(modname)
            return False
        except Exception:
            return True

    if not _broken("scipy.sparse") and not _broken("scipy.optimize"):
        return

    def _mod(name, package=False):
        module = types.ModuleType(name)
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=package)
        if package:
            module.__path__ = []
            spec.submodule_search_locations = []
        module.__spec__ = spec
        module.__file__ = f"<cloak-stub:{name}>"
        sys.modules[name] = module
        return module

    def _unavailable(*_args, **_kwargs):
        raise RuntimeError("scipy/sklearn is stubbed because scipy is unusable on this Python")

    scipy = sys.modules.get("scipy")
    if scipy is None or getattr(scipy, "__spec__", None) is None:
        scipy = _mod("scipy", package=True)

    sparse = _mod("scipy.sparse", package=True)
    optimize = _mod("scipy.optimize", package=True)
    stats = _mod("scipy.stats", package=True)
    sparse.issparse = lambda *_a, **_k: False
    sparse.csr_array = _unavailable
    sparse.csr_matrix = _unavailable
    optimize.linear_sum_assignment = _unavailable
    stats.fisher_exact = _unavailable
    scipy.sparse = sparse
    scipy.optimize = optimize
    scipy.stats = stats

    sklearn = _mod("sklearn", package=True)
    metrics = _mod("sklearn.metrics", package=True)
    metrics.roc_curve = _unavailable
    sklearn.metrics = metrics
    print("scipy is broken; stubbed scipy/sklearn so BGE embed/rerank can load.")


_stub_sklearn_if_scipy_broken()


def _flag(name, default="0"):
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# Dedicated instruct model for answers. Vision stays on a separate model.
LLM_MODEL = os.getenv("CLOAK_LLM_MODEL", "qwen2.5:7b")
VISION_MODEL = os.getenv("CLOAK_VISION_MODEL", "qwen2.5vl:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", os.getenv("CLOAK_OLLAMA_URL", "http://127.0.0.1:11434")).rstrip("/")
ENABLE_VISION = _flag("CLOAK_VISION", "1")
OFFLINE = _flag("CLOAK_OFFLINE", "0")
EMBEDDING_MODEL_NAME = os.getenv("CLOAK_EMBEDDING_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("CLOAK_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# Names that are not useful as chat/completion models in the UI dropdown.
_EMBED_MODEL_MARKERS = (
    "embed",
    "embedding",
    "bge-",
    "e5-",
    "gte-",
    "minilm",
    "nomic-embed",
    "mxbai-embed",
    "snowflake-arctic-embed",
)


def is_chat_model_name(name: str) -> bool:
    text = (name or "").strip().lower()
    if not text:
        return False
    return not any(marker in text for marker in _EMBED_MODEL_MARKERS)


_OLLAMA_MODELS_CACHE = {"ts": 0.0, "payload": None}
_OLLAMA_MODELS_TTL = 30.0


def list_ollama_models(force=False):
    """Return installed Ollama model names suitable for chat, plus reachability."""
    import time
    import urllib.error
    import urllib.request

    now = time.time()
    cached = _OLLAMA_MODELS_CACHE.get("payload")
    if (
        not force
        and cached is not None
        and (now - float(_OLLAMA_MODELS_CACHE.get("ts") or 0)) < _OLLAMA_MODELS_TTL
    ):
        return cached

    url = f"{OLLAMA_BASE_URL}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        result = {
            "ok": False,
            "error": str(exc),
            "models": [],
            "default": LLM_MODEL,
            "ollama_url": OLLAMA_BASE_URL,
        }
        _OLLAMA_MODELS_CACHE["ts"] = now
        _OLLAMA_MODELS_CACHE["payload"] = result
        return result

    names = []
    for item in payload.get("models") or []:
        name = (item.get("name") or item.get("model") or "").strip()
        if name and is_chat_model_name(name):
            names.append(name)
    # Prefer short tags without duplicate :latest clutter when both exist.
    deduped = []
    seen = set()
    for name in sorted(names):
        key = name[:-7] if name.endswith(":latest") else name
        if key in seen:
            continue
        seen.add(key)
        deduped.append(name)

    default = LLM_MODEL if LLM_MODEL in deduped else (deduped[0] if deduped else LLM_MODEL)
    result = {
        "ok": True,
        "error": None,
        "models": deduped,
        "default": default,
        "ollama_url": OLLAMA_BASE_URL,
    }
    _OLLAMA_MODELS_CACHE["ts"] = now
    _OLLAMA_MODELS_CACHE["payload"] = result
    return result


# Fetch wide so the reranker can promote a buried hit; keep TOP_DOCS after scoring.
RETRIEVE_K = _int("CLOAK_RETRIEVE_K", 50)
RERANK_CANDIDATES = _int("CLOAK_RERANK_CANDIDATES", 50)
TOP_DOCS = _int("CLOAK_TOP_DOCS", 5)
MAX_PACKED_DOCS = _int("CLOAK_MAX_PACKED_DOCS", 8)
EXTRACT_TOP_DOCS = _int("CLOAK_EXTRACT_TOP_DOCS", 5)
COMPARE_TOP_PER_SIDE = _int("CLOAK_COMPARE_TOP_PER_SIDE", 2)
SUMMARIZE_MAP_CHUNKS = _int("CLOAK_SUMMARIZE_MAP_CHUNKS", 16)
AGENT_RETRIES = _int("CLOAK_AGENT_RETRIES", 1)

# MiniLM/BGE logits mapped through sigmoid. Below this, skip generation
# unless a retry produces a passing hit.
RERANK_MIN_SCORE = _float("CLOAK_RERANK_MIN_SCORE", 0.20)
RERANK_MAX_CHARS = _int("CLOAK_RERANK_MAX_CHARS", 1800)

NUM_CTX = _int("CLOAK_NUM_CTX", 16384)
NUM_PREDICT = _int("CLOAK_NUM_PREDICT", 4096)
PROMPT_RESERVE_TOKENS = _int("CLOAK_PROMPT_RESERVE_TOKENS", 400)
# Parent page text sent to the LLM (small-to-big).
MAX_CHUNK_CHARS = _int("CLOAK_MAX_CHUNK_CHARS", 4000)
PARENT_TEXT_CHARS = _int("CLOAK_PARENT_TEXT_CHARS", 3200)
# 0 = fill leftover window automatically. Set a positive cap to override.
MAX_CONTEXT_CHARS = _int("CLOAK_MAX_CONTEXT_CHARS", 16000)
HISTORY_MESSAGES = _int("CLOAK_HISTORY_MESSAGES", 8)
CHARS_PER_TOKEN = 3.5

# auto = try hi_res, then ocr_only, then fast.
PDF_PARSE_STRATEGY = os.getenv("CLOAK_PDF_STRATEGY", "auto").strip().lower() or "auto"
CHILD_CHUNK_SIZE = _int("CLOAK_CHILD_CHUNK_SIZE", 480)
CHILD_CHUNK_OVERLAP = _int("CLOAK_CHILD_CHUNK_OVERLAP", 80)

# Back-compat for modules still importing the previous names.
LLM_MODEL = LLM_MODEL
NUM_CTX = NUM_CTX
NUM_PREDICT = NUM_PREDICT
PROMPT_RESERVE_TOKENS = PROMPT_RESERVE_TOKENS
MAX_CHUNK_CHARS = MAX_CHUNK_CHARS
MAX_PACKED_DOCS = MAX_PACKED_DOCS
EXTRACT_TOP_DOCS = EXTRACT_TOP_DOCS
COMPARE_TOP_PER_SIDE = COMPARE_TOP_PER_SIDE
SUMMARIZE_MAP_CHUNKS = SUMMARIZE_MAP_CHUNKS
AGENT_RETRIES = AGENT_RETRIES
RERANK_CANDIDATES = RERANK_CANDIDATES
RERANK_MAX_CHARS = RERANK_MAX_CHARS
RERANK_MIN_SCORE = RERANK_MIN_SCORE
HISTORY_MESSAGES = HISTORY_MESSAGES
CHARS_PER_TOKEN = CHARS_PER_TOKEN
CHILD_CHUNK_SIZE = CHILD_CHUNK_SIZE
CHILD_CHUNK_OVERLAP = CHILD_CHUNK_OVERLAP
EMBEDDING_MODEL_NAME = EMBEDDING_MODEL_NAME
ENABLE_VISION = ENABLE_VISION
PARENT_TEXT_CHARS = PARENT_TEXT_CHARS
PDF_PARSE_STRATEGY = PDF_PARSE_STRATEGY
RETRIEVE_K = RETRIEVE_K
VISION_MODEL = VISION_MODEL

if OFFLINE:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


@lru_cache(maxsize=1)
def torch_device():
    """Pick a device that can actually run kernels, not just one CUDA reports."""
    forced = os.getenv("CLOAK_DEVICE", "").strip().lower()
    if forced == "cpu":
        print("CLOAK_DEVICE=cpu; embeddings and rerank stay on CPU.")
        return "cpu"
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
        left = torch.randn(32, 32, device="cuda")
        right = torch.randn(32, 32, device="cuda")
        _ = (left @ right).cpu()
        torch.cuda.synchronize()
        print(f"CUDA usable for embeddings/rerank: {torch.cuda.get_device_name(0)}")
        return "cuda"
    except Exception as exc:
        print(
            f"CUDA cannot run this PyTorch build ({exc}). "
            "Embeddings and rerank will use CPU. Ollama can still use the GPU."
        )
        return "cpu"


def describe():
    return (
        f"Cloak local RAG  llm={LLM_MODEL}  vision={VISION_MODEL}  "
        f"embeddings={EMBEDDING_MODEL_NAME}  rerank={RERANK_MODEL}  "
        f"device={torch_device()}  ctx={NUM_CTX}  top={TOP_DOCS}  "
        f"rerank_min={RERANK_MIN_SCORE}  pdf={PDF_PARSE_STRATEGY}  "
        f"vision={'on' if ENABLE_VISION else 'off'}  "
        f"offline={'on' if OFFLINE else 'off'}"
    )
