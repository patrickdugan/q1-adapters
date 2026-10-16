# Phase B (torch): train one Bonsai LoRA per env, export GGUF. Waits for a free GPU.
. "$PSScriptRoot\common.ps1"
$ErrorActionPreference = 'Continue'
$W = 'C:\Users\patri\Documents\Codex\models\q1-adapters'
$jobs = @(
    @{ env = 'reverse-text'; data = "$R\runs\sft\reverse-text-gold.jsonl"; extra = @('--min-score', '0.99', '--epochs', '3', '--lr', '2e-4', '--max-len', '512', '--batch', '8', '--grad-accum', '2') },
    @{ env = 'gsm8k'; data = "$R\runs\rft\gsm8k-q1-base.jsonl"; extra = @('--min-score', '1.0', '--per-prompt', '2', '--epochs', '2', '--lr', '1e-4', '--max-len', '1536', '--batch', '4', '--grad-accum', '4') },
    @{ env = 'medmcqa'; data = "$R\runs\rft\medmcqa-q1-base.jsonl"; extra = @('--min-score', '1.0', '--per-prompt', '2', '--epochs', '2', '--lr', '1e-4', '--max-len', '2048', '--batch', '2', '--grad-accum', '8') },
    @{ env = 'ifeval'; data = "$R\runs\rft\ifeval-q1-base.jsonl"; extra = @('--min-score', '1.0', '--per-prompt', '2', '--epochs', '2', '--lr', '1e-4', '--max-len', '3072', '--batch', '1', '--grad-accum', '16') }
)
foreach ($j in $jobs) {
    $out = Join-Path $W $j.env
    if (Test-Path (Join-Path $out 'lora-f16.gguf')) { Log "phase B: $($j.env) adapter exists, skip"; continue }
    if (-not (Test-Path $j.data)) { Log "phase B: $($j.env) data missing ($($j.data)), skip"; continue }
    WaitGpu 11000
    Log "phase B: train $($j.env)"
    $t0 = Get-Date
    $a = @('-X', 'utf8', "$R\harness\train_lora.py", '--env', $j.env, '--data', $j.data, '--out', $out, '--log', "$R\runs\train\$($j.env).json") + $j.extra
    & $TorchPy @a *> "$R\runs\train\$($j.env).log"
    Log ("phase B: {0} exit={1} ({2:n0}s): {3}" -f $j.env, $LASTEXITCODE, ((Get-Date) - $t0).TotalSeconds, ((Get-Content "$R\runs\train\$($j.env).log" | Select-String -Pattern 'train items|saved adapter|gguf:|Error' | ForEach-Object { $_.Line.Trim() }) -join ' | '))
}
Log 'phase B: done'
