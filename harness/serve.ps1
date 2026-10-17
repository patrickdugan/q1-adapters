# Start (or restart) a detached llama-server for one configuration and wait until healthy.
#   serve.ps1 -Name q1 -Model <gguf> [-Lora <gguf>,<gguf>] [-Port 8096] [-Np 8] [-Ctx 16384]
# Uses the PrismML llama.cpp fork for every model (Q1_0 needs it; Q4_K_M runs on it too), with the
# Qwen3 chat template from the Bonsai unpacked snapshot so all models see byte-identical prompts.
# Records pid, args, VRAM before/after and model+adapter file sizes to runs/servers/<Name>.json.
param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string]$Model,
    [string[]]$Lora = @(),
    [int]$Port = 8096,
    [int]$Np = 8,
    [int]$Ctx = 16384,
    [switch]$LoraInitWithoutApply
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$srvDir = Join-Path $repo 'runs\servers'
New-Item -ItemType Directory -Force $srvDir | Out-Null
$exe = 'C:\Users\patri\Documents\Codex\BitAgent-gym\llama-prism\bin\llama-server.exe'
$tmpl = 'C:\Users\patri\Documents\Codex\BitAgent-gym\runs\hf-cache\models--prism-ml--Bonsai-8B-unpacked\snapshots\376f381570d6115bc03f82adcfa4af0c7672ae54\chat_template.jinja'

# stop only a server this script started on the same port
Get-ChildItem $srvDir -Filter '*.json' | ForEach-Object {
    $s = Get-Content $_.FullName | ConvertFrom-Json
    if ($s.port -eq $Port -and $s.pid) {
        $p = Get-Process -Id $s.pid -ErrorAction SilentlyContinue
        if ($p -and $p.ProcessName -eq 'llama-server') { Stop-Process -Id $s.pid -Confirm:$false; Start-Sleep -Seconds 3 }
    }
}
function UsedMiB { [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) }
$before = UsedMiB
$argList = @('-m', $Model, '--port', "$Port", '--host', '127.0.0.1', '-ngl', '99', '-c', "$Ctx", '--parallel', "$Np",
             '--jinja', '--chat-template-file', $tmpl, '--metrics',
             '--cache-ram', '1024')  # host prompt cache defaults to 8 GiB of RAM; the eval prompts barely reuse prefixes
foreach ($l in $Lora) { $argList += @('--lora', $l) }
if ($LoraInitWithoutApply) { $argList += '--lora-init-without-apply' }
$log = Join-Path $srvDir "$Name.log"
$p = Start-Process -FilePath $exe -ArgumentList $argList -RedirectStandardOutput "$log.out" -RedirectStandardError $log -WindowStyle Hidden -PassThru
$ok = $false
for ($i = 0; $i -lt 120; $i++) {
    Start-Sleep -Seconds 2
    if ($p.HasExited) { break }
    try { if ((Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2).status -eq 'ok') { $ok = $true; break } } catch {}
}
Start-Sleep -Seconds 2
$after = UsedMiB
$sizes = @{ model_bytes = (Get-Item $Model).Length; lora_bytes = @($Lora | ForEach-Object { (Get-Item $_).Length }) }
$rec = [ordered]@{ name = $Name; pid = $p.Id; port = $Port; ok = $ok; model = $Model; lora = $Lora; np = $Np; ctx = $Ctx;
                   vram_before_mib = $before; vram_after_mib = $after; vram_delta_mib = $after - $before;
                   model_bytes = $sizes.model_bytes; lora_bytes = $sizes.lora_bytes; started = (Get-Date).ToString('s') }
$rec | ConvertTo-Json | Set-Content (Join-Path $srvDir "$Name.json")
if (-not $ok) { Get-Content $log -Tail 20; throw "server $Name failed to start" }
"server $Name pid=$($p.Id) port=$Port vram_delta=$($after - $before) MiB"
