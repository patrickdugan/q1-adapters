# Auto-resume after a crash or forced restart (run at logon by the "q1-adapters resume" scheduled task; see
# install_resume_task.ps1). Relaunches every phase that left a runs\active\<name>.json marker and is not running.
# Phases resume per item, so a relaunch only redoes what the crash interrupted. Never stops anything.
. "$PSScriptRoot\common.ps1"
$ErrorActionPreference = 'Continue'
$d = Join-Path $R 'runs\active'
$markers = @(Get-ChildItem $d -Filter '*.json' -ErrorAction SilentlyContinue)
if (-not $markers.Count) { exit 0 }

# after a GPU bugcheck the driver can take a while to come back: wait (up to 10 min) for nvidia-smi to answer
for ($i = 0; $i -lt 20; $i++) {
    $null = nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0) { break }
    Start-Sleep -Seconds 30
}
Start-Sleep -Seconds 60  # let other logon work (OneDrive, the user's apps) settle first

$ps = @(Get-CimInstance Win32_Process -Filter "name='powershell.exe' or name='pwsh.exe'" | Where-Object { $_.ProcessId -ne $PID })
foreach ($mf in $markers) {
    $m = Get-Content $mf.FullName -Raw | ConvertFrom-Json
    if (-not (Test-Path $m.script)) { Log "resume: $($mf.Name): script $($m.script) missing, marker left"; continue }
    $leaf = Split-Path -Leaf $m.script
    if ($ps | Where-Object { $_.CommandLine -and $_.CommandLine.Contains($leaf) }) { Log "resume: $leaf already running"; continue }
    $a = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $m.script) + @($m.args | Where-Object { $_ })
    # a logon task starts with a fresh environment: restore the original session's tags (children inherit them) and
    # label the job, so GPU memory it takes is still attributable to the session that started the phase
    if ($m.env) { $m.env.PSObject.Properties | ForEach-Object { [Environment]::SetEnvironmentVariable($_.Name, [string]$_.Value) } }
    if (-not $env:VRAMORAMA_OWNER) { $env:VRAMORAMA_OWNER = "q1-adapters resume: $leaf" }
    $base = Join-Path $R ("runs\" + [IO.Path]::GetFileNameWithoutExtension($leaf))
    $p = Start-Process powershell.exe -ArgumentList $a -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput "$base.resume.out.log" -RedirectStandardError "$base.resume.err.log"
    Log "resume: relaunched $leaf (pid $($p.Id); marker from pid $($m.pid), started $($m.started))"
}
