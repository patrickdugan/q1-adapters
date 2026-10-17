# Phase D (4-bit server): Qwen3-8B Q4_K_M base, + online adapters, + our Bonsai adapters (reverse transfer);
# then llama-bench speed for Q1 vs Q4 on GPU and CPU.
. "$PSScriptRoot\common.ps1"
$ErrorActionPreference = 'Continue'
$W = 'C:\Users\patri\Documents\Codex\models\q1-adapters'
$envs = @('reverse-text', 'gsm8k', 'medmcqa', 'ifeval')
$names = @($Online.Keys)
$loras = @($Online.Values)
$ours = @($envs | Where-Object { Test-Path "$W\$_\lora-f16.gguf" })
$names += @($ours | ForEach-Object { "ours-$_" })
$loras += @($ours | ForEach-Object { "$W\$_\lora-f16.gguf" })
WaitGpu 11000
Log 'phase D: start Q4 server with online + our adapters (init without apply)'
& "$R\harness\serve.ps1" -Name q4 -Model $Q4 -Lora $loras -Port 8096 -Np 4 -Ctx 32768 -LoraInitWithoutApply | ForEach-Object { Log $_ }
$off = @(for ($i = 0; $i -lt $names.Count; $i++) { 0.0 })
Log ("lora scales: " + (SetLora 8096 $off))
foreach ($e in $envs) { Gen $e 'q4-base' "$R\results\$e\q4-base" }
$plan = @(@('gsm8k', 'gsm8k-grpo'), @('gsm8k', 'gsm8k-metamath'), @('medmcqa', 'medmcqa-vireshk'), @('ifeval', 'ifeval-tulu'))
foreach ($e in $ours) { $plan += , @($e, "ours-$e") }
foreach ($p in $plan) {
    $scales = @(for ($i = 0; $i -lt $names.Count; $i++) { if ($names[$i] -eq $p[1]) { 1.0 } else { 0.0 } })
    Log ("lora scales: " + (SetLora 8096 $scales) + " for $($p[1])")
    $lab = if ($p[1] -like 'ours-*') { 'q4+bonsai-adapter' } else { "q4+online:$($p[1])" }
    $stem = if ($p[1] -like 'ours-*') { 'q4+bonsai-adapter' } else { "q4+online-$($p[1])" }
    Gen $p[0] $lab "$R\results\$($p[0])\$stem"
}
StopServer 8096

# memory-matched 4-bit baseline: Qwen3-1.7B Q4_K_M (1.11 GB, about the size of Bonsai Q1_0)
$Q4s = 'C:\Users\patri\Documents\Codex\models\gguf\Qwen3-1.7B-Q4_K_M.gguf'
if (Test-Path $Q4s) {
    Log 'phase D: start Qwen3-1.7B Q4_K_M server'
    & "$R\harness\serve.ps1" -Name q4-1.7b -Model $Q4s -Port 8096 -Np 4 -Ctx 32768 | ForEach-Object { Log $_ }
    foreach ($e in $envs) { Gen $e 'q4-1.7b-base' "$R\results\$e\q4-1.7b-base" }
    StopServer 8096
}

Log 'phase D: llama-bench Q1 vs Q4 (GPU, then CPU)'
$bench = 'C:\Users\patri\Documents\Codex\BitAgent-gym\llama-prism\bin\llama-bench.exe'
New-Item -ItemType Directory -Force "$R\results\speed" | Out-Null
foreach ($m in @(@('q1', $Q1), @('q4', $Q4), @('q4-1.7b', $Q4s))) {
    & $bench -m $m[1] -ngl 99 -p 512 -n 128 -r 3 -o json 2> "$R\results\speed\$($m[0])-gpu.err" | Set-Content "$R\results\speed\$($m[0])-gpu.json"
    & $bench -m $m[1] -ngl 0 -p 128 -n 32 -r 2 -t 8 -o json 2> "$R\results\speed\$($m[0])-cpu.err" | Set-Content "$R\results\speed\$($m[0])-cpu.json"
    Log "phase D: bench $($m[0]) done"
}
Log 'phase D: done'
