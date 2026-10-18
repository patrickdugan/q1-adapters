# Phase Z, remainder: the server wedged after a GPU reset (all slots "processing", GPU idle). Restart and finish:
# Tulu IF on our Q4_K_M (resumes from 86/200), then our Bonsai adapters on it. A run that ends incomplete
# gets the server restarted and one more attempt.
. "$PSScriptRoot\common.ps1"
$ErrorActionPreference = 'Continue'
$Q4own = 'C:\Users\patri\Documents\Codex\models\gguf\phaseY\Q4_K_M.gguf'
$W = 'C:\Users\patri\Documents\Codex\models\q1-adapters'
$envs = @('reverse-text', 'gsm8k', 'medmcqa', 'ifeval')
$names = @($Online.Keys) + @($envs | ForEach-Object { "ours-$_" })
$loras = @($Online.Values) + @($envs | ForEach-Object { "$W\$_\lora-f16.gguf" })
function Only([string]$n) { @(for ($i = 0; $i -lt $names.Count; $i++) { if ($names[$i] -eq $n) { 1.0 } else { 0.0 } }) }
function Start-Q4own { WaitGpu 11000; & "$R\harness\serve.ps1" -Name q4own2 -Model $Q4own -Lora $loras -Port 8096 -Np 4 -Ctx 32768 -LoraInitWithoutApply | ForEach-Object { Log $_ } }
function Rows([string]$e, [string]$stem) { $f = "$R\results\$e\$stem.jsonl"; if (Test-Path $f) { (Get-Content $f | Measure-Object -Line).Lines } else { 0 } }

Start-Q4own
$plan = @(@('ifeval', 'ifeval-tulu')) + @($envs | ForEach-Object { , @($_, "ours-$_") })
foreach ($pl in $plan) {
    if ($pl[1] -like 'ours-*') { $lab = 'q4own+bonsai-adapter'; $stem = 'q4own+bonsai-adapter' }
    else { $lab = "q4own+online:$($pl[1])"; $stem = "q4own+online-$($pl[1])" }
    for ($try = 0; $try -lt 2; $try++) {
        Log ("lora scales: " + (SetLora 8096 (Only $pl[1])) + " for $($pl[1]) on q4own (try $try)")
        Gen $pl[0] $lab "$R\results\$($pl[0])\$stem"
        if ((Rows $pl[0] $stem) -ge 200) { break }
        Log "phase Z2: $stem incomplete ($(Rows $pl[0] $stem)/200) -- restarting server"
        StopServer 8096
        Start-Q4own
    }
}
StopServer 8096
& 'C:\Users\patri\OneDrive\Documents\GitHub\Spiral\.venv\Scripts\python.exe' -X utf8 "$PSScriptRoot\make_tables.py" *> (Join-Path $R 'runs\tables.out.log')
Log 'phase Z2: done'
