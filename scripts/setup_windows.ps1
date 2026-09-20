#requires -Version 7.0

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
[Console]::InputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$env:PYTHONUTF8 = '1'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'Visual Studio C++ tools are required.' }
. (Join-Path $vs 'Common7\Tools\Launch-VsDevShell.ps1') -Arch amd64 -HostArch amd64 -SkipAutomaticLocation | Out-Null

$nvcc = Get-Command nvcc.exe -ErrorAction Stop
$env:CUDA_HOME = Split-Path -Parent (Split-Path -Parent $nvcc.Source)
$env:DISTUTILS_USE_SDK = '1'
$env:MSSdk = '1'
$env:TORCH_DONT_CHECK_COMPILER_ABI = '1'
$env:NVCC_PREPEND_FLAGS = '-allow-unsupported-compiler'
if (-not $env:MAX_JOBS) { $env:MAX_JOBS = '4' }
$eigen = Join-Path $repo 'submodules\gaussianhierarchy\dependencies\eigen'
$env:INCLUDE = "$eigen;$env:INCLUDE"

$python = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { uv venv --python 3.10 --seed .venv }
& $python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
& $python -m pip install -r requirements.txt 'gsplat==1.5.3'
$env:TORCH_CUDA_ARCH_LIST = (& nvidia-smi --query-gpu=compute_cap --format=csv,noheader | Select-Object -First 1).Trim()
& $python -m pip install submodules/simple-knn submodules/gaussianhierarchy --no-build-isolation
& $python -m pip install 'git+https://github.com/rahul-goel/fused-ssim/' --no-build-isolation
& $python -c 'from utils.gsplat_compat import prepare_gsplat_windows; prepare_gsplat_windows(); from gsplat.cuda._backend import _C; print("gsplat CUDA backend ready")'
cmake -G 'Ninja Multi-Config' -S submodules/gaussianhierarchy -B submodules/gaussianhierarchy/build '-DCMAKE_POLICY_VERSION_MINIMUM=3.5' '-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler'
cmake --build submodules/gaussianhierarchy/build --config Release --parallel
