# Shared helpers for the phase scripts.
$Global:R = Split-Path -Parent $PSScriptRoot
$Global:VfPy = 'C:\Users\patri\.venvs\vf\Scripts\python.exe'
$Global:TorchPy = 'C:\Users\patri\Documents\Codex\BitAgent-gym\.venv\Scripts\python.exe'
$Global:Q1 = 'C:\Users\patri\Documents\Codex\BitAgent-gym\prime-artifacts\Bonsai-8B-Q1_0.gguf'
$Global:Q4 = 'C:\Users\patri\Documents\Codex\models\gguf\Qwen3-8B-Q4_K_M.gguf'
$Global:AD = 'C:\Users\patri\Documents\Codex\models\adapters'
$Global:Online = [ordered]@{
    'gsm8k-grpo'  = "$AD\Pradheep1647__qwen3_8b_gsm8k_unloth_trl\lora-f16.gguf"
    'gsm8k-metamath' = "$AD\tianzl66__Qwen3-8B-MetaMathQA-50K-LoRA\lora-f16.gguf"
    'medmcqa-vireshk' = "$AD\vireshk__qwen3-8b-medmcqa-qlora\lora-f16.gguf"
    'ifeval-tulu' = "$AD\tianzl66__Qwen3-8B-InstructionFollowing-LoRA\lora-f16.gguf"
}
$env:HF_HUB_OFFLINE = '0'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:PYTHONIOENCODING = 'utf-8'

function Log([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $msg
    $line
    for ($i = 0; $i -lt 20; $i++) {  # a reader (e.g. a tail) may hold the file briefly; retry rather than lose the line
        try { Add-Content (Join-Path $R 'runs\pipeline.log') $line -ErrorAction Stop; return } catch { Start-Sleep -Milliseconds 500 }
    }
}

function WaitGpu([int]$needMiB = 9000) {
    while ($true) {
        $q = (nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits) -split ','
        $free = [int]$q[1] - [int]$q[0]
        if ($free -ge $needMiB) { return }
        Log "waiting for GPU: $free MiB free, need $needMiB"
        Start-Sleep -Seconds 120
    }
}

# Set LoRA scales on a running server: $Scales is an array aligned with the --lora order (0 = off).
function SetLora([int]$Port, [double[]]$Scales) {
    $body = @(for ($i = 0; $i -lt $Scales.Count; $i++) { @{ id = $i; scale = $Scales[$i] } })
    $json = ConvertTo-Json -InputObject $body -Compress
    Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$Port/lora-adapters" -Body $json -ContentType 'application/json' | Out-Null
    $now = Invoke-RestMethod "http://127.0.0.1:$Port/lora-adapters"
    ($now | ForEach-Object { "{0}:{1}" -f $_.id, $_.scale }) -join ' '
}

function Gen([string]$EnvName, [string]$Label, [string]$Out, [string[]]$Extra = @(), [int]$Port = 8096) {
    $args2 = @('-X', 'utf8', "$R\harness\generate.py", '--env', $EnvName, '--label', $Label, '--out', $Out,
               '--base-url', "http://127.0.0.1:$Port/v1", '--concurrency', '6', '--resume') + $Extra
    $t0 = Get-Date
    $o = & $VfPy @args2 2>&1 | Where-Object { $_ -match '^\{|Traceback|Error:' }
    Log ("{0} {1}: {2} ({3:n0}s)" -f $EnvName, $Label, ($o -join ' '), ((Get-Date) - $t0).TotalSeconds)
}

function StopServer([int]$Port = 8096) {
    Get-ChildItem (Join-Path $R 'runs\servers') -Filter '*.json' -ErrorAction SilentlyContinue | ForEach-Object {
        $s = Get-Content $_.FullName | ConvertFrom-Json
        if ($s.port -eq $Port -and $s.pid) {
            $p = Get-Process -Id $s.pid -ErrorAction SilentlyContinue
            if ($p -and $p.ProcessName -eq 'llama-server') { Stop-Process -Id $s.pid -Confirm:$false; Start-Sleep -Seconds 3 }
        }
    }
}
