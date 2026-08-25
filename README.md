# Cloak

Local-first RAG chat: FastAPI backend, web UI, and optional Electron desktop shell.

## Project layout

```text
Cloak/
├── server.py                 # FastAPI app entry (uvicorn server:app)
├── accounts.py               # Auth & per-user PDF storage
├── harness.py                # Chat system prompt & direct-answer helpers
├── chat_tools.py             # Calculator, clock, PDF tools, etc.
├── history_aware_generation.py
├── ingestion_pipeline.py
├── route.py
├── session_manager.py
├── cloak_config.py
│
├── web/                      # Browser UI (served at /)
├── desktop/                  # Electron app (npm start)
├── assets/                   # Shared logos & icons
├── legacy/                   # Original CustomTkinter UI (CloakUI.py)
├── scripts/                  # Utility scripts (e.g. network smoke test)
├── tests/                    # Unit tests
├── docs/                     # Documentation (CRAG setup, etc.)
├── samples/                  # Local test PDFs (gitignored when large)
│
├── eval_crag.py              # CRAG benchmark harness
├── crag_rescore.py
├── run_crag_*.py|sh|ps1      # CRAG run helpers
├── setup_crag_*.sh|ps1|bat
└── crag_results/             # CRAG JSON outputs
```

## Quick start

### Backend

```powershell
cd Cloak
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn server:app --host 127.0.0.1 --port 8001
```

Open [http://127.0.0.1:8001](http://127.0.0.1:8001).

### Desktop (optional)

```powershell
cd Cloak\desktop
npm install
npm start
```

The desktop app expects the backend at `http://127.0.0.1:8001` by default.

### Tests

```powershell
cd Cloak
python -m unittest discover -s tests -v
```

## Assets

| File | Used by |
|------|---------|
| `assets/logo.png` | Canonical brand logo |
| `web/logo.png` | Web UI (static) |
| `desktop/icon.png` | Electron window icon |
| `assets/cloaklogo.*` | Legacy CustomTkinter UI |

After updating `assets/logo.png`, copy to `web/logo.png` and `desktop/icon.png`.

## CRAG evaluation

See [docs/CRAG_5070.md](docs/CRAG_5070.md).
