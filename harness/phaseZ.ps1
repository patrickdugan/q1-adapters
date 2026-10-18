# Phase Z: the fair Q4 + community-adapter comparison. Qwen's official Qwen3-8B-Q4_K_M.gguf is built from an
# "awq-compatible" rescaled checkpoint (general.name/finetune metadata; norm weights differ 20-55%), so LoRAs trained
# on Qwen/Qwen3-8B land in the wrong basis there. Here: our own Q4_K_M converted from the exact Qwen/Qwen3-8B weights
# (phase Y), base + every online adapter + our Bonsai adapters, all four envs, 200 items, env sampling (as the main table).
. "$PSScriptRoot\common.ps1"
$ErrorActionPreference = 'Continue'
$mark = (Get-Content (Join-Path $R 'runs\pipeline.log') | Measure-Object -Line).Lines
while ($true) {
    $tail = Get-Content (Join-Path $R 'runs\pipeline.log') | Select-Object -Skip $mark
    if ($tail -match 'phase Y3 \(cpu\): done') { break }
    Start-Sleep -Seconds 60
}
$Q4own = 'C:\Users\patri\Documents\Codex\models\gguf\phaseY\Q4_K_M.gguf'
$W = 'C:\Users\patri\Documents\Codex\models\q1-adapters'
$envs = @('reverse-text', 'gsm8k', 'medmcqa', 'ifeval')
$names = @($Online.Keys) + @($envs | ForEach-Object { "ours-$_" })
$loras = @($Online.Values) + @($envs | ForEach-Object { "$W\$_\lora-f16.gguf" })
function Only([string]$n) { @(for ($i = 0; $i -lt $names.Count; $i++) { if ($names[$i] -eq $n) { 1.0 } else { 0.0 } }) }

WaitGpu 11000
Log 'phase Z: start own-Q4_K_M server (exact Qwen/Qwen3-8B basis) with online + our adapters'
& "$R\harness\serve.ps1" -Name q4own -Model $Q4own -Lora $loras -Port 8096 -Np 4 -Ctx 32768 -LoraInitWithoutApply | ForEach-Object { Log $_ }
SetLora 8096 (Only '') | Out-Null
foreach ($e in $envs) { Gen $e 'q4own-base' "$R\results\$e\q4own-base" }
$plan = @(@('gsm8k', 'gsm8k-grpo'), @('gsm8k', 'gsm8k-metamath'), @('medmcqa', 'medmcqa-vireshk'), @('ifeval', 'ifeval-tulu'))
foreach ($e in $envs) { $plan += , @($e, "ours-$e") }
foreach ($pl in $plan) {
    Log ("lora scales: " + (SetLora 8096 (Only $pl[1])) + " for $($pl[1]) on q4own")
    if ($pl[1] -like 'ours-*') { $lab = 'q4own+bonsai-adapter'; $stem = 'q4own+bonsai-adapter' }
    else { $lab = "q4own+online:$($pl[1])"; $stem = "q4own+online-$($pl[1])" }
    Gen $pl[0] $lab "$R\results\$($pl[0])\$stem"
}
StopServer 8096
& 'C:\Users\patri\OneDrive\Documents\GitHub\Spiral\.venv\Scripts\python.exe' -X utf8 "$PSScriptRoot\make_tables.py" *> (Join-Path $R 'runs\tables.out.log')
Log 'phase Z: done'
