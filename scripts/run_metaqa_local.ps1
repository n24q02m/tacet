#!/usr/bin/env pwsh
# Local MetaQA cascade benchmark — runs entirely on the local machine,
# no Modal / GPU dependency.  See ``scripts/run_grok_local.ps1`` for the same
# xAI key sourcing convention.  Run from the repository root.

param(
    [int]$Hop = 1,
    [int]$Limit = 100,
    [string]$Out = "experiments/results/metaqa_local_grok.json",
    [string]$ApiKey = $env:XAI_API_KEY,
    [string]$Model = "grok-4.3"
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

Write-Host "[$([DateTime]::Now.ToString('HH:mm:ss'))] MetaQA hop=$Hop limit=$Limit (Grok $Model teacher)"
.venv/Scripts/python.exe -u experiments/run_metaqa.py --metaqa-root data/MetaQA --hop $Hop --limit $Limit --out $Out
