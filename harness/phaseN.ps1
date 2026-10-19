# Phase N: more Hub environments and bigger samples, on Q1 (Bonsai 8B), our correct-basis Q4_K_M 8B and
# Qwen3-1.7B Q4_K_M. Staged so the core comparison lands first on a shared, power-capped GPU:
#   stage 1  new envs, base models: math500 500, aime 60 x k=2, pubmedqa 500, reasoning-core 500, ascii-tree 200
#   stage 2  bigger n, base models: gsm8k full test (1319), medmcqa 1000; plus our GSM8K/MedMCQA adapters on Q1
#   stage 3  our GSM8K adapter on math500 (does it transfer to harder math?)
# A run that ends short of its expected rows gets the server restarted and one more attempt (TDR wedges).
. "$PSScriptRoot\common.ps1"
$ErrorActionPreference = 'Continue'
$Q4own = 'C:\Users\patri\Documents\Codex\models\gguf\phaseY\Q4_K_M.gguf'
$Q4s = 'C:\Users\patri\Documents\Codex\models\gguf\Qwen3-1.7B-Q4_K_M.gguf'
$up = 'C:\Users\patri\Documents\Codex\2026-09-30\pu\work\llama.cpp\b11269\llama-server.exe'
$W = 'C:\Users\patri\Documents\Codex\models\q1-adapters'
$envs4 = @('reverse-text', 'gsm8k', 'medmcqa', 'ifeval')
$ours = @($envs4 | ForEach-Object { "$W\$_\lora-f16.gguf" })
function Only([string]$n) { @(for ($i = 0; $i -lt $envs4.Count; $i++) { if ("ours-$($envs4[$i])" -eq $n) { 1.0 } else { 0.0 } }) }

$New = [ordered]@{ 'math500' = 500; 'aime' = 120; 'pubmedqa' = 500; 'reasoning-core' = 500; 'ascii-tree' = 200 }
$Big = [ordered]@{ 'gsm8k' = 1319; 'medmcqa' = 1000 }
function Rows([string]$e, [string]$stem) { $f = "$R\results\$e\$stem.jsonl"; if (Test-Path $f) { (Get-Content $f | Measure-Object -Line).Lines } else { 0 } }
function XArgs([string]$e) {
    if ($e -eq 'aime') { return @('--k', '2') }
    if ($Big.Contains($e)) { return @('--n', [string]$Big[$e]) }
    return @()
}
function Expect([string]$e) { if ($Big.Contains($e)) { $Big[$e] } else { $New[$e] } }

$Start = @{
    'q1'    = { WaitGpu 7000; & "$R\harness\serve.ps1" -Name n-q1 -Model $Q1 -Lora $ours -Port 8096 -Np 4 -Ctx 32768 -LoraInitWithoutApply | ForEach-Object { Log $_ } }
    'q4own' = { WaitGpu 9000; & "$R\harness\serve.ps1" -Name n-q4own -Model $Q4own -Port 8096 -Np 4 -Ctx 32768 | ForEach-Object { Log $_ } }
    # the PrismML fork's CUDA path crashes on Qwen3-1.7B: upstream llama.cpp
    'q4s'   = { WaitGpu 6000; & "$R\harness\serve.ps1" -Name n-q4-1.7b -Model $Q4s -Port 8096 -Np 4 -Ctx 32768 -Exe $up | ForEach-Object { Log $_ } }
}
$Stem = @{ 'q1' = 'q1-base'; 'q4own' = 'q4own-base'; 'q4s' = 'q4-1.7b-base' }

# Run one (env, label, stem) on the current server; restart and retry once if short.
function Run([string]$m, [string]$e, [string]$lab, [string]$stem, [double[]]$Scales = $null) {
    for ($try = 0; $try -lt 2; $try++) {
        if ((Rows $e $stem) -ge (Expect $e)) { return }
        if ($Scales) { Log ("lora scales: " + (SetLora 8096 $Scales) + " for $stem/$e") }
        Gen $e $lab "$R\results\$e\$stem" (XArgs $e)
        $have = Rows $e $stem; $want = Expect $e
        if ($have -ge $want) { return }
        Log "phase N: $e/$stem incomplete ($have/$want) -- restarting server (try $try)"
        StopServer 8096
        & $Start[$m]
    }
}
function Leg([string]$m, [string[]]$envList) {
    Log "phase N: $m on $($envList -join ', ')"
    & $Start[$m]
    foreach ($e in $envList) { Run $m $e $Stem[$m] $Stem[$m] $(if ($m -eq 'q1') { Only '' } else { $null }) }
    if ($m -eq 'q1' -and $envList -contains 'gsm8k') {
        Run $m 'gsm8k' 'q1+ours' 'q1+ours' (Only 'ours-gsm8k')
        Run $m 'medmcqa' 'q1+ours' 'q1+ours' (Only 'ours-medmcqa')
    }
    StopServer 8096
}

foreach ($m in @('q1', 'q4s', 'q4own')) { Leg $m @($New.Keys) }
& 'C:\Users\patri\OneDrive\Documents\GitHub\Spiral\.venv\Scripts\python.exe' -X utf8 "$PSScriptRoot\make_tables.py" *> (Join-Path $R 'runs\tables.out.log')
Log 'phase N: stage 1 done'
foreach ($m in @('q1', 'q4s', 'q4own')) { Leg $m @($Big.Keys) }
& 'C:\Users\patri\OneDrive\Documents\GitHub\Spiral\.venv\Scripts\python.exe' -X utf8 "$PSScriptRoot\make_tables.py" *> (Join-Path $R 'runs\tables.out.log')
Log 'phase N: stage 2 done'
& $Start['q1']
Run 'q1' 'math500' 'q1+ours:gsm8k' 'q1+ours-gsm8k' (Only 'ours-gsm8k')
StopServer 8096
& 'C:\Users\patri\OneDrive\Documents\GitHub\Spiral\.venv\Scripts\python.exe' -X utf8 "$PSScriptRoot\make_tables.py" *> (Join-Path $R 'runs\tables.out.log')
Log 'phase N: done'
