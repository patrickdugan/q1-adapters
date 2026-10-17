# Wait for the B-C-D chain to finish, then run phase E.
. "$PSScriptRoot\common.ps1"
$mark = (Get-Content (Join-Path $R 'runs\pipeline.log') | Measure-Object -Line).Lines
while ($true) {
    $tail = Get-Content (Join-Path $R 'runs\pipeline.log') | Select-Object -Skip $mark
    if ($tail -match 'chain: all phases done') { break }
    Start-Sleep -Seconds 60
}
& "$PSScriptRoot\phaseE.ps1"
