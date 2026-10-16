# Phase A (1-bit server): base evals, online-adapter transfer evals, RFT sampling for training.
. "$PSScriptRoot\common.ps1"
$ErrorActionPreference = 'Continue'
$envs = @('reverse-text', 'gsm8k', 'medmcqa', 'ifeval')
$names = @($Online.Keys)
$loras = @($Online.Values)

WaitGpu 9000
Log 'phase A: start Q1 server with online adapters (init without apply)'
& "$R\harness\serve.ps1" -Name q1-online -Model $Q1 -Lora $loras -Port 8096 -Np 6 -Ctx 49152 -LoraInitWithoutApply | ForEach-Object { Log $_ }

# 1) 1-bit base
Log ("lora scales: " + (SetLora 8096 @(0, 0, 0, 0)))
foreach ($e in $envs) { Gen $e 'q1-base' "$R\results\$e\q1-base" }

# 2) online Qwen3-8B adapters applied to the 1-bit base (transfer)
$plan = @(@('gsm8k', 'gsm8k-grpo'), @('gsm8k', 'gsm8k-metamath'), @('medmcqa', 'medmcqa-vireshk'), @('ifeval', 'ifeval-tulu'))
foreach ($p in $plan) {
    $scales = @(for ($i = 0; $i -lt $names.Count; $i++) { if ($names[$i] -eq $p[1]) { 1.0 } else { 0.0 } })
    Log ("lora scales: " + (SetLora 8096 $scales) + " for $($p[1])")
    Gen $p[0] "q1+online:$($p[1])" "$R\results\$($p[0])\q1+online-$($p[1])"
}

# 3) rejection-sampling data for our adapters (base model, k=4 on 400 train prompts)
Log ("lora scales: " + (SetLora 8096 @(0, 0, 0, 0)))
foreach ($e in @('gsm8k', 'medmcqa', 'ifeval')) {
    Gen $e 'q1-base-rft' "$R\runs\rft\$e-q1-base" @('--split', 'train', '--n', '400', '--k', '4', '--seed', '1')
}
StopServer 8096
Log 'phase A: done'
