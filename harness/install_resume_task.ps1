# Register (or replace) the per-user logon task that runs resume.ps1. No elevation needed; runs only when this user
# logs on, hidden, with a 2-minute delay. Remove with: Unregister-ScheduledTask -TaskName 'q1-adapters resume'
$resume = Join-Path $PSScriptRoot 'resume.ps1'
$user = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$resume`""
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
$trigger.Delay = 'PT2M'
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName 'q1-adapters resume' -Description 'Relaunch q1-adapters phases interrupted by a crash or restart (harness\resume.ps1)' `
    -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Select-Object TaskName, State
