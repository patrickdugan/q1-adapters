# Interim 1-bit work while another job shares the GPU (~6 GB footprint, port 8097, 4 slots x 8k):
# finish/extend the cross-task matrix to 200 per pair (routing end-to-end), then phase E's Q1 part (n=500).
. "$PSScriptRoot\common.ps1"
$ErrorActionPreference = 'Continue'
$W = 'C:\Users\patri\Documents\Codex\models\q1-adapters'
$envs = @('reverse-text', 'gsm8k', 'medmcqa', 'ifeval')
$names = @($Online.Keys) + @($envs | ForEach-Object { "ours-$_" })
$loras = @($Online.Values) + @($envs | ForEach-Object { "$W\$_\lora-f16.gguf" })
function Only([string]$n) { @(for ($i = 0; $i -lt $names.Count; $i++) { if ($names[$i] -eq $n) { 1.0 } else { 0.0 } }) }
$QPort = 8097
function G2([string]$e, [string]$lab, [string]$stem, [string[]]$x = @()) { Gen $e $lab "$R\results\$e\$stem" $x $QPort }

WaitGpu 7000
Log 'phase Q1x: start Q1 server (shared GPU: 4 slots, port 8097)'
& "$R\harness\serve.ps1" -Name q1-x -Model $Q1 -Lora $loras -Port $QPort -Np 4 -Ctx 32768 -LoraInitWithoutApply | ForEach-Object { Log $_ }

# 1) cross-task matrix: every adapter on every other env, 200 examples (resume extends the 100 already done)
foreach ($a in $envs) {
    SetLora $QPort (Only "ours-$a") | Out-Null
    foreach ($e in $envs) { if ($e -ne $a) { G2 $e "q1+ours:$a" "q1+ours-$a" @('--n', '200') } }
}
# 2) phase E's 1-bit share: gsm8k + medmcqa to 500
$N = @('--n', '500')
SetLora $QPort (Only '') | Out-Null
foreach ($e in @('gsm8k', 'medmcqa')) { G2 $e 'q1-base' 'q1-base' $N }
foreach ($p in @(@('gsm8k', 'gsm8k-grpo'), @('gsm8k', 'gsm8k-metamath'), @('medmcqa', 'medmcqa-vireshk'))) {
    SetLora $QPort (Only $p[1]) | Out-Null
    G2 $p[0] "q1+online:$($p[1])" "q1+online-$($p[1])" $N
}
foreach ($e in @('gsm8k', 'medmcqa')) { SetLora $QPort (Only "ours-$e") | Out-Null; G2 $e 'q1+ours' 'q1+ours' $N }
StopServer $QPort
Log 'phase Q1x: done'
