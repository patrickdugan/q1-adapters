# Wait for phase A to finish, then run B (train), C (Q1 + ours), D (Q4 baselines + speed), then tables.
. "$PSScriptRoot\common.ps1"
while (-not (Select-String -Path (Join-Path $R 'runs\pipeline.log') -Pattern 'phase A: done' -Quiet)) { Start-Sleep -Seconds 60 }
New-Item -ItemType Directory -Force (Join-Path $R 'runs\train') | Out-Null
& "$PSScriptRoot\phaseB.ps1"
& "$PSScriptRoot\phaseC.ps1"
& "$PSScriptRoot\phaseD.ps1"
& 'C:\Users\patri\OneDrive\Documents\GitHub\Spiral\.venv\Scripts\python.exe' -X utf8 "$PSScriptRoot\make_tables.py" *> (Join-Path $R 'runs\tables.out.log')
Log 'chain: all phases done'
