# Phase E: tighten the confidence intervals where the eval split allows it -- extend gsm8k and medmcqa
# from 200 to 500 examples for every main-table configuration (resume appends ids 200..499 to the same files).
. "$PSScriptRoot\common.ps1"
$ErrorActionPreference = 'Continue'
$W = 'C:\Users\patri\Documents\Codex\models\q1-adapters'
$N = @('--n', '500')
$names = @($Online.Keys)
$loras = @($Online.Values)
foreach ($e in @('gsm8k', 'medmcqa')) { if (Test-Path "$W\$e\lora-f16.gguf") { $names += "ours-$e"; $loras += "$W\$e\lora-f16.gguf" } }
function Only([string]$n) { @(for ($i = 0; $i -lt $names.Count; $i++) { if ($names[$i] -eq $n) { 1.0 } else { 0.0 } }) }

# first diagnose the Q4 + community-adapter collapse (phase X), then extend
& "$PSScriptRoot\phaseX.ps1"

foreach ($srv in @(@('q1', $Q1), @('q4', $Q4))) {
    $tag = $srv[0]
    WaitGpu 11000
    Log "phase E: start $tag server"
    & "$R\harness\serve.ps1" -Name "$tag-ext" -Model $srv[1] -Lora $loras -Port 8096 -Np 4 -Ctx 32768 -LoraInitWithoutApply | ForEach-Object { Log $_ }
    SetLora 8096 (Only '') | Out-Null
    foreach ($e in @('gsm8k', 'medmcqa')) { Gen $e "$tag-base" "$R\results\$e\$tag-base" $N }
    $plan = @(@('gsm8k', 'gsm8k-grpo'), @('gsm8k', 'gsm8k-metamath'), @('medmcqa', 'medmcqa-vireshk'), @('gsm8k', 'ours-gsm8k'), @('medmcqa', 'ours-medmcqa'))
    # Q4 + adapters collapse by 0.3-0.7 (see phase X), far beyond any CI question: extend only the Q4 base
    if ($tag -eq 'q4') { $plan = @() }
    foreach ($p in $plan) {
        if ($names -notcontains $p[1]) { continue }
        Log ("lora scales: " + (SetLora 8096 (Only $p[1])) + " for $($p[1]) on $tag")
        if ($p[1] -like 'ours-*') {
            $lab = if ($tag -eq 'q1') { 'q1+ours' } else { 'q4+bonsai-adapter' }
            $stem = $lab
        } else { $lab = "${tag}+online:$($p[1])"; $stem = "${tag}+online-$($p[1])" }
        Gen $p[0] $lab "$R\results\$($p[0])\$stem" $N
    }
    StopServer 8096
}
$Q4s = 'C:\Users\patri\Documents\Codex\models\gguf\Qwen3-1.7B-Q4_K_M.gguf'
WaitGpu 9000
& "$R\harness\serve.ps1" -Name q4-1.7b-ext -Model $Q4s -Port 8096 -Np 4 -Ctx 32768 | ForEach-Object { Log $_ }
foreach ($e in @('gsm8k', 'medmcqa')) { Gen $e 'q4-1.7b-base' "$R\results\$e\q4-1.7b-base" $N }
StopServer 8096
& 'C:\Users\patri\OneDrive\Documents\GitHub\Spiral\.venv\Scripts\python.exe' -X utf8 "$PSScriptRoot\make_tables.py" *> (Join-Path $R 'runs\tables.out.log')
Log 'phase E: done'
& "$PSScriptRoot\phaseM.ps1"
