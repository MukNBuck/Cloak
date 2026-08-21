# One-time setup for CRAG eval on the RTX 5070 Windows box.
# Run in PowerShell from the Cloak folder:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\setup_crag_5070.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Py = if ($env:PYTHON) { $env:PYTHON } else { "py" }
$Venv = if ($env:CLOAK_VENV) { $env:CLOAK_VENV } else { Join-Path $Root ".venv" }
$LlmModel = if ($env:CLOAK_LLM_MODEL) { $env:CLOAK_LLM_MODEL } else { "qwen2.5:14b" }
$TorchIndex = if ($env:TORCH_INDEX) { $env:TORCH_INDEX } else { "https://download.pytorch.org/whl/cu124" }

function Test-OllamaApi {
    try {
        $null = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" -UseBasicParsing -TimeoutSec 2
        return $true
    } catch {
        return $false
    }
}

function Install-OllamaWindows {
    if (Get-Command ollama -ErrorAction SilentlyContinue) {
        Write-Host "==> Ollama already on PATH: $((Get-Command ollama).Source)"
        return
    }
    Write-Host "==> Installing Ollama for Windows"
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        winget install --id Ollama.Ollama -e --accept-package-agreements --accept-source-agreements
    } else {
        $installer = Join-Path $env:TEMP "OllamaSetup.exe"
        Write-Host "==> Downloading OllamaSetup.exe"
        Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $installer
        Write-Host "==> Running installer (complete the wizard if it appears)"
        Start-Process -FilePath $installer -Wait
    }
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path", "User")
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        $guess = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
        if (Test-Path $guess) {
            $env:Path = "$($env:LOCALAPPDATA)\Programs\Ollama;$env:Path"
        }
    }
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        throw "Ollama installed but not on PATH. Open a new PowerShell and re-run, or add Ollama to PATH."
    }
}

function Start-OllamaWindows {
    if (Test-OllamaApi) {
        Write-Host "==> Ollama already responding on :11434"
        return
    }
    Write-Host "==> Starting Ollama"
    # App install usually registers a tray app / service; start the CLI serve as fallback.
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    for ($i = 0; $i -lt 45; $i++) {
        if (Test-OllamaApi) {
            Write-Host "==> Ollama is up"
            return
        }
        Start-Sleep -Seconds 1
    }
    throw "Ollama did not start. Launch Ollama from the Start menu, then re-run this script."
}

Write-Host "==> Python"
& $Py -V

if (-not (Test-Path $Venv)) {
    Write-Host "==> Creating venv at $Venv"
    & $Py -3 -m venv $Venv
}

$Activate = Join-Path $Venv "Scripts\Activate.ps1"
. $Activate
python -m pip install -U pip wheel

Write-Host "==> Installing CUDA PyTorch ($TorchIndex)"
try {
    python -m pip install --index-url $TorchIndex torch torchvision torchaudio
} catch {
    Write-Host "CUDA wheel install failed; trying default torch"
    python -m pip install torch
}

Write-Host "==> Installing Cloak deps"
python -m pip install -r requirements.txt

$EnvFile = Join-Path $Root ".env"
$EnvExample = Join-Path $Root ".env.example"
if (-not (Test-Path $EnvFile) -and (Test-Path $EnvExample)) {
    Copy-Item $EnvExample $EnvFile
    Write-Host "==> Wrote .env from .env.example — edit CLOAK_CRAG_PATH"
}

Write-Host "==> GPU check"
python -c "import torch; print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

Install-OllamaWindows
Start-OllamaWindows

Write-Host "==> Pulling Ollama models"
ollama pull $LlmModel
try { ollama pull qwen2.5:7b } catch { Write-Host "optional 7b pull skipped" }

Write-Host ""
Write-Host "Setup done."
Write-Host "  1. Edit .env and set CLOAK_CRAG_PATH to your jsonl"
Write-Host "  2. .\run_crag_5070.ps1 smoke"
Write-Host "See CRAG_5070.md"
