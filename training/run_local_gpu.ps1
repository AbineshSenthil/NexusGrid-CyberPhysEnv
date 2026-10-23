# NexusGrid - local GPU pipeline for Windows (tested on an RTX 4060 Laptop GPU, 8 GB).
#
#   1. evaluate the base model as a pure operator on held-out seeds
#   2. GRPO fine-tuning with simulator rewards (LoRA on a 4-bit model)
#   3. evaluate the trained adapter on the same held-out seeds
#
# Usage (from the repository root, in PowerShell):
#   powershell -ExecutionPolicy Bypass -File training\run_local_gpu.ps1            # full run
#   powershell -ExecutionPolicy Bypass -File training\run_local_gpu.ps1 -Smoke     # 2-step dry run
#
# Logs are written as UTF-8 to runs\grpo\*.log, results to results\llm\.
param(
    [string]$Model = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
    [string]$Backend = "unsloth",
    [int]$Steps = 200,
    [int]$EvalSeeds = 5,
    [switch]$Smoke,
    [switch]$SkipBaseline
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
New-Item -ItemType Directory -Force -Path "runs\grpo" | Out-Null

function Run-Step([string]$Name, [string[]]$Arguments) {
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    & $Py @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
}

Run-Step "GPU check" @("-c", "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')")

if ($Smoke) {
    Run-Step "GRPO smoke test" @("training\train_grpo.py", "--backend", $Backend, "--model", $Model, "--load-in-4bit",
        "--max-steps", "2", "--dataset-episodes", "5", "--num-generations", "2", "--grad-accum", "1",
        "--out", "runs\grpo\smoke", "--log", "runs\grpo\smoke.log")
    Run-Step "Eval smoke test" @("training\eval_llm_local.py", "--backend", $Backend, "--model", $Model,
        "--tasks", "0", "--seeds", "1", "--out", "results\llm\smoke", "--log", "runs\grpo\smoke_eval.log")
    Write-Host "`nSmoke test finished." -ForegroundColor Green
    exit 0
}

if (-not $SkipBaseline) {
    Run-Step "Baseline evaluation" @("training\eval_llm_local.py", "--backend", $Backend, "--model", $Model,
        "--seeds", "$EvalSeeds", "--out", "results\llm\qwen3-1.7b-base", "--log", "runs\grpo\eval_base.log")
}
Run-Step "GRPO training" @("training\train_grpo.py", "--backend", $Backend, "--model", $Model, "--load-in-4bit",
    "--max-steps", "$Steps", "--out", "runs\grpo\qwen3-1.7b", "--log", "runs\grpo\train.log")
Run-Step "Evaluation after GRPO" @("training\eval_llm_local.py", "--backend", $Backend, "--model", $Model,
    "--adapter", "runs\grpo\qwen3-1.7b\adapter", "--seeds", "$EvalSeeds", "--out", "results\llm\qwen3-1.7b-grpo",
    "--log", "runs\grpo\eval_grpo.log")
Write-Host "`nAll done. Results in results\llm\, training curve in runs\grpo\qwen3-1.7b\reward_curve.png" -ForegroundColor Green
