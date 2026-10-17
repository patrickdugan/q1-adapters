# After chainF: finish the Q4 medmcqa n=500 extension, then the transformers+PEFT reference check.
. "$PSScriptRoot\common.ps1"
$mark = (Get-Content (Join-Path $R 'runs\pipeline.log') | Measure-Object -Line).Lines
while ($true) {
    $tail = Get-Content (Join-Path $R 'runs\pipeline.log') | Select-Object -Skip $mark
    if ($tail -match 'chainF: done') { break }
    Start-Sleep -Seconds 60
}
for ($try = 0; $try -lt 3; $try++) {  # the fork has crashed on TDR resets; resume until complete
    $n = (Get-Content "$R\results\medmcqa\q4-base.jsonl" | Measure-Object -Line).Lines
    if ($n -ge 500) { break }
    WaitGpu 12000
    & "$R\harness\serve.ps1" -Name "q4-ext3-$try" -Model $Q4 -Port 8096 -Np 4 -Ctx 32768 | ForEach-Object { Log $_ }
    Gen 'medmcqa' 'q4-base' "$R\results\medmcqa\q4-base" @('--n', '500')
    StopServer 8096
}
# transformers + PEFT on real Qwen3-8B (NF4) -- wait for the download to finish
$snap = Get-ChildItem 'C:\Users\patri\Documents\Codex\models\hf\models--Qwen--Qwen3-8B\snapshots' -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
while (-not $snap -or (@(Get-ChildItem $snap.FullName -Filter 'model-0000*-of-00005.safetensors').Count -lt 5) -or
       (Get-Process -Id ((Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*dl_qwen3_8b*' }).ProcessId) -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 60
    $snap = Get-ChildItem 'C:\Users\patri\Documents\Codex\models\hf\models--Qwen--Qwen3-8B\snapshots' -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
}
$P = "$R\runs\peft"
$mm = "$AD\tianzl66__Qwen3-8B-MetaMathQA-50K-LoRA"
$grpo = "$AD\Pradheep1647__qwen3_8b_gsm8k_unloth_trl"
foreach ($cfg in @(@('none', 'nf4-base'), @($mm, 'nf4-metamath'), @($grpo, 'nf4-grpo'))) {
    WaitGpu 10000
    & $TorchPy -X utf8 "$R\harness\peft_check.py" generate --prompts "$P\gsm8k.prompts.jsonl" --adapter $cfg[0] --out "$P\gsm8k.$($cfg[1]).jsonl" *> "$P\gsm8k.$($cfg[1]).log"
    $s = & $VfPy -X utf8 "$R\harness\peft_check.py" score --env gsm8k --file "$P\gsm8k.$($cfg[1]).jsonl" 2>&1 | Where-Object { $_ -match '^\{' }
    Log "peft check gsm8k $($cfg[1]): $s"
}
& 'C:\Users\patri\OneDrive\Documents\GitHub\Spiral\.venv\Scripts\python.exe' -X utf8 "$PSScriptRoot\make_tables.py" *> (Join-Path $R 'runs\tables.out.log')
Log 'chainG: done'
