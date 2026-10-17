# Phase M: deployment footprint. For each base: a 1-slot, 4k-context server with no adapters, then with
# four adapters loaded; record VRAM delta (nvidia-smi) and the server's resident RAM. Writes results/memory.json.
. "$PSScriptRoot\common.ps1"
$ErrorActionPreference = 'Continue'
$W = 'C:\Users\patri\Documents\Codex\models\q1-adapters'
$ours = @('reverse-text', 'gsm8k', 'medmcqa', 'ifeval') | ForEach-Object { "$W\$_\lora-f16.gguf" }
$Q4s = 'C:\Users\patri\Documents\Codex\models\gguf\Qwen3-1.7B-Q4_K_M.gguf'
$out = @()
foreach ($m in @(@('q1-bonsai-8b', $Q1), @('q4-qwen3-8b', $Q4), @('q4-qwen3-1.7b', $Q4s))) {
    foreach ($withLora in @($false, $true)) {
        if ($withLora -and $m[0] -eq 'q4-qwen3-1.7b') { continue }  # 8B-shaped adapters do not fit the 1.7B
        WaitGpu 12000
        $name = "mem-$($m[0])-$(if ($withLora) { 'lora4' } else { 'nolora' })"
        $l = if ($withLora) { $ours } else { @() }
        & "$R\harness\serve.ps1" -Name $name -Model $m[1] -Lora $l -Port 8098 -Np 1 -Ctx 4096 | Out-Null
        Start-Sleep -Seconds 5
        $rec = Get-Content "$R\runs\servers\$name.json" | ConvertFrom-Json
        $ws = (Get-Process -Id $rec.pid -ErrorAction SilentlyContinue).WorkingSet64
        $out += [ordered]@{ config = $name; model_bytes = $rec.model_bytes; lora_bytes = ($rec.lora_bytes | Measure-Object -Sum).Sum;
                            vram_delta_mib = $rec.vram_delta_mib; ram_working_set_mib = [math]::Round($ws / 1MB) }
        StopServer 8098
        Log "phase M: $name vram_delta=$($rec.vram_delta_mib) MiB ram=$([math]::Round($ws/1MB)) MiB"
    }
}
$out | ConvertTo-Json | Set-Content "$R\results\memory.json"
Log 'phase M: done'
