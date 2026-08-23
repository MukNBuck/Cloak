# Run CRAG eval presets on the Windows 5070 box.
# Usage (from Cloak/):
#   .\run_crag_5070.ps1 smoke
#   .\run_crag_5070.ps1 corpus
#   .\run_crag_5070.ps1 retrieval
#   .\run_crag_5070.ps1 generate
#   .\run_crag_5070.ps1 all

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$EnvFile = Join-Path $Root ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#")) { return }
        $parts = $line.Split("=", 2)
        if ($parts.Count -eq 2) {
            $name = $parts[0].Trim()
            $value = $parts[1].Trim().Trim('"')
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

$Venv = if ($env:CLOAK_VENV) { $env:CLOAK_VENV } else { Join-Path $Root ".venv" }
$Activate = Join-Path $Venv "Scripts\Activate.ps1"
if (Test-Path $Activate) { . $Activate }

if (-not $env:CLOAK_VISION) { $env:CLOAK_VISION = "0" }
if (-not $env:CLOAK_LLM_MODEL) { $env:CLOAK_LLM_MODEL = "qwen2.5:14b" }

# Ensure ollama is reachable for generate mode
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    $guess = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
    if (Test-Path (Join-Path $guess "ollama.exe")) {
        $env:Path = "$guess;$env:Path"
    }
}

$OutDir = if ($env:CLOAK_CRAG_OUT) { $env:CLOAK_CRAG_OUT } else { Join-Path $Root "crag_results" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ModeArg = if ($args.Count -ge 1) { $args[0] } else { "smoke" }
$Limit = if ($env:LIMIT) { [int]$env:LIMIT } else { $null }

function Invoke-Crag {
    param(
        [string]$Name,
        [string[]]$EvalArgs
    )
    $out = Join-Path $OutDir ("{0}_{1}.json" -f $Name, $Stamp)
    Write-Host ""
    Write-Host "======== $Name ========"
    Write-Host ("cmd: python eval_crag.py " + ($EvalArgs -join " "))
    Write-Host "out: $out"
    python eval_crag.py @EvalArgs --out $out
}

switch ($ModeArg) {
    "smoke" {
        Invoke-Crag smoke_corpus @("--mode", "corpus", "--limit", "80", "--html")
        Invoke-Crag smoke_retrieval @("--mode", "retrieval", "--limit", "20", "--html")
    }
    "corpus" {
        $n = if ($Limit) { $Limit } else { 200 }
        Invoke-Crag corpus @("--mode", "corpus", "--limit", "$n", "--html")
    }
    "retrieval" {
        $n = if ($Limit) { $Limit } else { 100 }
        Invoke-Crag retrieval @("--mode", "retrieval", "--limit", "$n", "--html")
    }
    "generate" {
        $n = if ($Limit) { $Limit } else { 40 }
        Invoke-Crag generate @("--mode", "generate", "--limit", "$n", "--html")
    }
    "all" {
        Invoke-Crag corpus @("--mode", "corpus", "--limit", "200", "--html")
        Invoke-Crag retrieval @("--mode", "retrieval", "--limit", "100", "--html")
        Invoke-Crag generate @("--mode", "generate", "--limit", "40", "--html")
    }
    default {
        Write-Host "Unknown mode: $ModeArg"
        Write-Host "Use: smoke | corpus | retrieval | generate | all"
        exit 1
    }
}

Write-Host ""
Write-Host "Results in $OutDir"
