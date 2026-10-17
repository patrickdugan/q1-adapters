# Wait for phase E (+ M) to finish, then run the phase X diagnosis and rebuild the tables.
. "$PSScriptRoot\common.ps1"
$mark = (Get-Content (Join-Path $R 'runs\pipeline.log') | Measure-Object -Line).Lines
while ($true) {
    $tail = Get-Content (Join-Path $R 'runs\pipeline.log') | Select-Object -Skip $mark
    if ($tail -match 'phase M: done') { break }
    Start-Sleep -Seconds 60
}
while (-not (Test-Path 'C:\Users\patri\Documents\Codex\models\gguf\Qwen3-8B-Q8_0.gguf')) { Start-Sleep -Seconds 60 }
& "$PSScriptRoot\phaseX.ps1"
& 'C:\Users\patri\OneDrive\Documents\GitHub\Spiral\.venv\Scripts\python.exe' -X utf8 "$PSScriptRoot\make_tables.py" *> (Join-Path $R 'runs\tables.out.log')
Log 'chainX: done'
