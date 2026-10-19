# Phase N2 (after phase N): three more Bonsai adapters trained on the env's own signal, scored on Q1 against
# phase N's q1-base rows --
#   ascii-tree      gold SFT, 800 train rows (a transformation skill, like reverse-text)
#   reasoning-core  gold SFT on the answer alone, no derivation, 800 rows (does a format-level adapter buy reasoning?)
#   pubmedqa        rejection sampling from Q1 base, 400 train prompts x 4 (every base model over-answers "maybe")
param([int]$WaitPid = 0)
. "$PSScriptRoot\common.ps1"
$ErrorActionPreference = 'Continue'
$W = 'C:\Users\patri\Documents\Codex\models\q1-adapters'
if ($WaitPid) { while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 60 } }
BeginPhase $PSCommandPath
# batch 1 x 16 (same effective batch as 2 x 8): at 2 x 8 the ascii-tree run filled the 16 GB card and spilled 2.4 GB
# into shared system memory
$gold = @('--min-score', '0.99', '--epochs', '3', '--lr', '2e-4', '--max-len', '2048', '--batch', '1', '--grad-accum', '16')
$rft = @('--min-score', '1.0', '--per-prompt', '2', '--epochs', '2', '--lr', '1e-4', '--max-len', '2048', '--batch', '1', '--grad-accum', '16')
$jobs = [ordered]@{
    'ascii-tree'     = @{ data = "$R\runs\sft\ascii-tree-gold.jsonl"; extra = $gold }
    'reasoning-core' = @{ data = "$R\runs\sft\reasoning-core-gold.jsonl"; extra = $gold }
    'pubmedqa'       = @{ data = "$R\runs\rft\pubmedqa-q1-base.jsonl"; extra = $rft }
}
New-Item -ItemType Directory -Force "$R\runs\train", "$R\runs\rft" | Out-Null

# RFT samples for pubmedqa (same recipe as phase A: 400 train prompts, k=4, seed 1)
if ((Test-Path $jobs['pubmedqa'].data) -and ((Get-Content $jobs['pubmedqa'].data | Measure-Object -Line).Lines -ge 1600)) { Log 'phase N2: pubmedqa RFT samples exist' }
else {
    WaitGpu 7000
    & "$R\harness\serve.ps1" -Name n2-rft -Model $Q1 -Port 8096 -Np 4 -Ctx 32768 | ForEach-Object { Log $_ }
    for ($try = 0; $try -lt 2; $try++) {
        Gen 'pubmedqa' 'q1-base-rft' "$R\runs\rft\pubmedqa-q1-base" @('--split', 'train', '--n', '400', '--k', '4', '--seed', '1')
        if ((Get-Content $jobs['pubmedqa'].data | Measure-Object -Line).Lines -ge 1600) { break }
        StopServer 8096
        & "$R\harness\serve.ps1" -Name n2-rft -Model $Q1 -Port 8096 -Np 4 -Ctx 32768 | ForEach-Object { Log $_ }
    }
    StopServer 8096
}

foreach ($e in $jobs.Keys) {
    $out = Join-Path $W $e
    if (GoodGguf (Join-Path $out 'lora-f16.gguf')) { Log "phase N2: $e adapter exists, skip"; continue }
    if (Test-Path $out) { Log "phase N2: $e adapter dir present but GGUF missing or corrupt -- retraining"; Remove-Item -Recurse -Force $out }
    WaitGpu 11000
    Log "phase N2: train $e"
    $t0 = Get-Date
    $a = @('-X', 'utf8', "$R\harness\train_lora.py", '--env', $e, '--data', $jobs[$e].data, '--out', $out,
           '--log', "$R\runs\train\$e.json") + $jobs[$e].extra
    & $TorchPy @a *> "$R\runs\train\$e.log"
    Log ("phase N2: {0} exit={1} ({2:n0}s): {3}" -f $e, $LASTEXITCODE, ((Get-Date) - $t0).TotalSeconds, ((Get-Content "$R\runs\train\$e.log" | Select-String -Pattern 'selected|saved adapter|gguf:|Error' | ForEach-Object { $_.Line.Trim() }) -join ' | '))
}

$new = @($jobs.Keys | Where-Object { GoodGguf "$W\$_\lora-f16.gguf" })
if ($new.Count -lt $jobs.Count) { Log "phase N2: missing adapters (have: $($new -join ', '))" }
if ($new.Count) {
    $loras = @($new | ForEach-Object { "$W\$_\lora-f16.gguf" })
    $start = { WaitGpu 7000; & "$R\harness\serve.ps1" -Name n2-q1 -Model $Q1 -Lora $loras -Port 8096 -Np 4 -Ctx 32768 -LoraInitWithoutApply | ForEach-Object { Log $_ } }
    & $start
    $want = @{ 'ascii-tree' = 200; 'reasoning-core' = 500; 'pubmedqa' = 500 }
    for ($i = 0; $i -lt $new.Count; $i++) {
        $e = $new[$i]; $f = "$R\results\$e\q1+ours.jsonl"
        $sc = @(for ($j = 0; $j -lt $new.Count; $j++) { if ($j -eq $i) { 1.0 } else { 0.0 } })
        for ($try = 0; $try -lt 2; $try++) {
            Log ("lora scales: " + (SetLora 8096 $sc) + " for ours-$e")
            Gen $e 'q1+ours' "$R\results\$e\q1+ours"
            if ((Test-Path $f) -and (Get-Content $f | Measure-Object -Line).Lines -ge $want[$e]) { break }
            Log "phase N2: $e/q1+ours incomplete -- restarting server"
            StopServer 8096
            & $start
        }
    }
    StopServer 8096
}
& 'C:\Users\patri\OneDrive\Documents\GitHub\Spiral\.venv\Scripts\python.exe' -X utf8 "$PSScriptRoot\make_tables.py" *> (Join-Path $R 'runs\tables.out.log')
Log 'phase N2: done'
EndPhase $PSCommandPath
