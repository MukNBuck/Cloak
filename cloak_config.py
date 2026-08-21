"""Local-only runtime settings for air-gapped NDA deployments.

Nothing here talks to the public internet. Ollama and Hugging Face models
must already be on disk; set CLOAK_OFFLINE=1 after the first cache.
"""

from functools import lru_cache
import os

from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


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


LLM_MODEL = os.getenv("CLOAK_LLM_MODEL", "llava")
VISION_MODEL = os.getenv("CLOAK_VISION_MODEL", LLM_MODEL)
ENABLE_VISION = _flag("CLOAK_VISION", "0")
OFFLINE = _flag("CLOAK_OFFLINE", "0")
EMBEDDING_MODEL_NAME = os.getenv("CLOAK_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("CLOAK_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
# Fetch wide so the reranker can promote a buried hit; keep only TOP_DOCS after scoring.
RETRIEVE_K = _int("CLOAK_RETRIEVE_K", 24)
RERANK_CANDIDATES = _int("CLOAK_RERANK_CANDIDATES", 24)
TOP_DOCS = _int("CLOAK_TOP_DOCS", 3)
MAX_PACKED_DOCS = _int("CLOAK_MAX_PACKED_DOCS", TOP_DOCS)
# MiniLM/BGE logits mapped through sigmoid. Below this, skip generation.
RERANK_MIN_SCORE = _float("CLOAK_RERANK_MIN_SCORE", 0.35)
# Cross-encoders typically truncate at 512 tokens; don't score a 4k chunk blindly.
RERANK_MAX_CHARS = _int("CLOAK_RERANK_MAX_CHARS", 1800)
NUM_CTX = _int("CLOAK_NUM_CTX", 4096)
NUM_PREDICT = _int("CLOAK_NUM_PREDICT", 384)
PROMPT_RESERVE_TOKENS = _int("CLOAK_PROMPT_RESERVE_TOKENS", 280)
MAX_CHUNK_CHARS = _int("CLOAK_MAX_CHUNK_CHARS", 900)
# 0 = fill leftover window automatically. Set a positive cap to override.
MAX_CONTEXT_CHARS = _int("CLOAK_MAX_CONTEXT_CHARS", 0)
HISTORY_MESSAGES = _int("CLOAK_HISTORY_MESSAGES", 8)
CHARS_PER_TOKEN = 3.5

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
        f"Cloak local RAG  llm={LLM_MODEL}  embeddings={EMBEDDING_MODEL_NAME}  "
        f"device={torch_device()}  ctx={NUM_CTX}  top={TOP_DOCS}  "
        f"rerank_min={RERANK_MIN_SCORE}  vision={'on' if ENABLE_VISION else 'off'}  "
        f"offline={'on' if OFFLINE else 'off'}"
    )
