# Phase X: why do community Qwen3-8B adapters break Qwen3-8B Q4_K_M in llama.cpp (gsm8k 0.945 -> 0.25/0.43,
# stutter) while the authors report e.g. 0.84 (bf16, vLLM) and the same adapters are neutral on Bonsai Q1?
# Grid: base weights {Q4_K_M, Q8_0} x adapter {none, MetaMathQA} x decoding {greedy, env sampling}, 100 gsm8k items;
# plus the Tulu IF adapter on ifeval for {Q4_K_M, Q8_0}; plus Bonsai Q1 + MetaMathQA greedy as reference.
. "$PSScriptRoot\common.ps1"
$ErrorActionPreference = 'Continue'
$Q8 = 'C:\Users\patri\Documents\Codex\models\gguf\Qwen3-8B-Q8_0.gguf'
$D = "$R\results\diagnostics"
$mm = $Online['gsm8k-metamath']; $tulu = $Online['ifeval-tulu']
$G = @('--n', '100', '--temperature', '0', '--top-p', '1', '--top-k', '0')
$S = @('--n', '100')
function Run([string]$name, [string]$model, [string[]]$lora, [string]$envn, [string]$tag, [string[]]$x) {
    WaitGpu 12000
    & "$R\harness\serve.ps1" -Name "x-$name" -Model $model -Lora $lora -Port 8096 -Np 4 -Ctx 32768 | ForEach-Object { Log $_ }
    Gen $envn "diag:$name:$tag" "$D\$envn\$name-$tag" $x
    StopServer 8096
}
foreach ($q in @(@('q4km', $Q4), @('q8', $Q8))) {
    if (-not (Test-Path $q[1])) { Log "phase X: missing $($q[1])"; continue }
    Run "$($q[0])-base" $q[1] @() 'gsm8k' 'greedy' $G
    Run "$($q[0])-metamath" $q[1] @($mm) 'gsm8k' 'greedy' $G
    Run "$($q[0])-metamath" $q[1] @($mm) 'gsm8k' 'sampled' $S
    Run "$($q[0])-tulu" $q[1] @($tulu) 'ifeval' 'sampled' $S
}
Run 'q1-metamath' $Q1 @($mm) 'gsm8k' 'greedy' $G
Log 'phase X: done'
