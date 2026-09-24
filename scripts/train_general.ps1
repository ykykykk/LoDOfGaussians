#requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Data,
    [Parameter(Mandatory)][string]$Output,
    [int]$Resolution = 2,
    [string]$Config = 'general_balanced.json',
    [int]$Iterations = 0,
    [int]$CoarseIterations = 0,
    [int]$Seed = 0,
    [string]$VcVarsAll = '',
    [string]$Toolset = '14.44',
    [switch]$PlanOnly,
    [switch]$SkipIfExists,
    [string]$ResumeCheckpoint = '',
    [switch]$AllowGrowthResume
)
$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$Python = Join-Path $Root '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw 'Project Python environment is missing.' }
if (-not (Test-Path -LiteralPath $Data)) { throw 'Dataset directory does not exist.' }
if (-not $PlanOnly) {
    if (-not $VcVarsAll) {
        $VsWhere = 'C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe'
        if (Test-Path -LiteralPath $VsWhere) {
            $VsRoot = & $VsWhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
            if ($VsRoot) { $VcVarsAll = Join-Path $VsRoot 'VC/Auxiliary/Build/vcvarsall.bat' }
        }
    }
    if (-not $VcVarsAll -or -not (Test-Path -LiteralPath $VcVarsAll)) { throw 'Pass -VcVarsAll with the installed vcvarsall.bat path.' }
    $Environment = & cmd.exe /d /s /c "`"$VcVarsAll`" x64 -vcvars_ver=$Toolset >nul && set"
    if ($LASTEXITCODE -ne 0) { throw 'MSVC initialization failed.' }
    foreach ($Line in $Environment) {
        $i = $Line.IndexOf('=')
        if ($i -gt 0) { Set-Item -LiteralPath "Env:$($Line.Substring(0,$i))" -Value $Line.Substring($i+1) }
    }
}
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONFAULTHANDLER = '1'
$env:DISTUTILS_USE_SDK = '1'
$env:MSSdk = '1'
$env:VSLANG = '1033'
$env:MAX_JOBS = '4'
$env:PYTHONDONTWRITEBYTECODE = '1'
$TrainArgs = @('train.py','--project_dir',$Data,'--output_dir',$Output,
    '--config',$Config,'--resolution',"$Resolution",'--seed',"$Seed",
    '--export_ply',(Join-Path $Output 'scene_finest.ply'))
if ($Iterations -gt 0) { $TrainArgs += @('--iterations',"$Iterations") }
if ($CoarseIterations -gt 0) { $TrainArgs += @('--coarse_iterations',"$CoarseIterations") }
if ($PlanOnly) { $TrainArgs += '--plan_only' }
if ($SkipIfExists) { $TrainArgs += '--skip_if_exists' }
if ($ResumeCheckpoint) { $TrainArgs += @('--resume_checkpoint', $ResumeCheckpoint) }
if ($AllowGrowthResume) { $TrainArgs += '--allow_growth_resume' }
& $Python @TrainArgs
if ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE" }
