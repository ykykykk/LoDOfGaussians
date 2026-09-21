#requires -Version 7.0
param([switch]$RequireCuda)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$Python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw 'Project .venv is missing.' }
if ($RequireCuda) {
    & $Python -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU required"; from utils.gsplat_compat import prepare_gsplat_windows; prepare_gsplat_windows(); from utils.resident_native import load_native; load_native("cuda"); print("Native CUDA backend compiled and loaded")'
}
& $Python -m pytest tests/test_streaming_v2.py tests/test_resident_pool.py tests/test_training_regressions.py -q
if ($LASTEXITCODE -ne 0) { throw 'Resident regression tests failed.' }
