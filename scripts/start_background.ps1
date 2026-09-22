#requires -Version 7.0
[CmdletBinding()]
param([Parameter(Mandatory)][string]$Runner)
$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$RunnerPath = (Resolve-Path -LiteralPath $Runner).Path
if ([IO.Path]::GetExtension($RunnerPath) -ne '.ps1' -or $RunnerPath.Contains('"')) { throw 'Runner must be a PowerShell script.' }
$TaskName = 'ALoD-' + [Guid]::NewGuid().ToString('N').Substring(0,12)
$Action = New-ScheduledTaskAction -Execute (Get-Command pwsh.exe).Source `
    -Argument ('-NoLogo -NoProfile -WindowStyle Hidden -File "' + $RunnerPath + '"') `
    -WorkingDirectory (Split-Path $PSScriptRoot -Parent)
$Principal = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $TaskName -Action $Action -Principal $Principal -Settings $Settings -Description 'ALoD training independent of the launching app lifecycle' | Out-Null
Start-ScheduledTask -TaskName $TaskName
[pscustomobject]@{TaskName=$TaskName; Runner=$RunnerPath}
