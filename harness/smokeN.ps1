# Smoke test for phase N's new envs: 4 examples each on Q1 base, into runs/smoke (gitignored).
. "$PSScriptRoot\common.ps1"
$ErrorActionPreference = 'Continue'
WaitGpu 7000
& "$R\harness\serve.ps1" -Name smoke-q1 -Model $Q1 -Port 8096 -Np 4 -Ctx 32768 | ForEach-Object { Log $_ }
foreach ($e in @('math500', 'aime', 'pubmedqa', 'reasoning-core', 'ascii-tree')) { Gen $e 'smoke' "$R\runs\smoke\$e" @('--n', '4') }
StopServer 8096
Log 'smoke N: done'
