# Phase C (1-bit server): Bonsai Q1 + our env adapters, all four loaded at once, switched per run.
. "$PSScriptRoot\common.ps1"
$ErrorActionPreference = 'Continue'
$W = 'C:\Users\patri\Documents\Codex\models\q1-adapters'
$envs = @('reverse-text', 'gsm8k', 'medmcqa', 'ifeval')
$have = @($envs | Where-Object { Test-Path "$W\$_\lora-f16.gguf" })
if (-not $have) { Log 'phase C: no adapters, skip'; return }
$loras = @($have | ForEach-Object { "$W\$_\lora-f16.gguf" })
WaitGpu 9000
Log "phase C: start Q1 server with our adapters: $($have -join ', ')"
& "$R\harness\serve.ps1" -Name q1-ours -Model $Q1 -Lora $loras -Port 8096 -Np 6 -Ctx 49152 -LoraInitWithoutApply | ForEach-Object { Log $_ }
for ($i = 0; $i -lt $have.Count; $i++) {
    $scales = @(for ($j = 0; $j -lt $have.Count; $j++) { if ($j -eq $i) { 1.0 } else { 0.0 } })
    Log ("lora scales: " + (SetLora 8096 $scales) + " for $($have[$i])")
    Gen $have[$i] 'q1+ours' "$R\results\$($have[$i])\q1+ours"
}
# cross-task check: each adapter on the other envs (does a task adapter hurt elsewhere?)
for ($i = 0; $i -lt $have.Count; $i++) {
    $scales = @(for ($j = 0; $j -lt $have.Count; $j++) { if ($j -eq $i) { 1.0 } else { 0.0 } })
    SetLora 8096 $scales | Out-Null
    foreach ($e in $envs) {
        if ($e -eq $have[$i]) { continue }
        Gen $e "q1+ours:$($have[$i])" "$R\results\$e\q1+ours-$($have[$i])" @('--n', '100')
    }
}
StopServer 8096
Log 'phase C: done'
