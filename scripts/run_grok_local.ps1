#!/usr/bin/env pwsh
# Local Grok 4.3 cascade benchmark — runs entirely on the local machine,
# no Modal / GPU dependency.  This script wires the env TACET expects and
# shells out to the existing ``experiments/run_real_llm_scale.py``.
#
# Provide the xAI key via:
#   1. ``$env:XAI_API_KEY`` — preferred (e.g. set in your shell profile)
#   2. CLI flag: ``./scripts/run_grok_local.ps1 -ApiKey xai-...``  (last resort)
#
# Run from the repository root (paths below are root-relative).

param(
    [int]$Limit = 200,
    [string]$Out = "experiments/results/grok_local.json",
    [string]$ApiKey = $env:XAI_API_KEY,
    [string]$Model = "grok-4.3",
    [int]$Seed = 0
)

if (-not $ApiKey) {
    Write-Error "Set `$env:XAI_API_KEY or pass -ApiKey xai-..."
    exit 1
}

$env:TACET_TEACHER         = "grok"
$env:TACET_XAI_API_KEY     = $ApiKey
$env:TACET_XAI_MODEL       = $Model
$env:GEMINI_API_KEY       = ""
$env:TACET_GEMINI_API_KEY  = ""
$env:TACET_KGE_BACKEND     = "numpy"
$env:TACET_KGE_EPOCHS      = "20"
$env:PYTHONUNBUFFERED     = "1"

Write-Host "[$([DateTime]::Now.ToString('HH:mm:ss'))] Grok $Model real_llm_scale limit=$Limit"
.venv/Scripts/python.exe -u experiments/run_real_llm_scale.py --limit $Limit --seed $Seed --out $Out
