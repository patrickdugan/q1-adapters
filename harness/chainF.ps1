# After phase X: rerun Qwen3-1.7B reverse-text on upstream llama.cpp (its first file mixed rows from the crashed
# fork run under a mangled name), then rebuild the tables.
. "$PSScriptRoot\common.ps1"
$mark = (Get-Content (Join-Path $R 'runs\pipeline.log') | Measure-Object -Line).Lines
while ($true) {
    $tail = Get-Content (Join-Path $R 'runs\pipeline.log') | Select-Object -Skip $mark
    if ($tail -match 'chainX: done') { break }
    Start-Sleep -Seconds 60
}
$up = 'C:\Users\patri\Documents\Codex\2026-09-30\pu\work\llama.cpp\b11269\llama-server.exe'
$Q4s = 'C:\Users\patri\Documents\Codex\models\gguf\Qwen3-1.7B-Q4_K_M.gguf'
WaitGpu 9000
& "$R\harness\serve.ps1" -Name q4-1.7b-rt -Model $Q4s -Port 8096 -Np 4 -Ctx 32768 -Exe $up | ForEach-Object { Log $_ }
Gen 'reverse-text' 'q4-1.7b-base' "$R\results\reverse-text\q4-1.7b-base"
StopServer 8096
& 'C:\Users\patri\OneDrive\Documents\GitHub\Spiral\.venv\Scripts\python.exe' -X utf8 "$PSScriptRoot\make_tables.py" *> (Join-Path $R 'runs\tables.out.log')
Log 'chainF: done'
