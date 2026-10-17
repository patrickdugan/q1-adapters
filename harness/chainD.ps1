# Restart of the chain from phase D (B and C are complete); releases chainE when done.
. "$PSScriptRoot\common.ps1"
& "$PSScriptRoot\phaseD.ps1"
& 'C:\Users\patri\OneDrive\Documents\GitHub\Spiral\.venv\Scripts\python.exe' -X utf8 "$PSScriptRoot\make_tables.py" *> (Join-Path $R 'runs\tables.out.log')
Log 'chain: all phases done'
