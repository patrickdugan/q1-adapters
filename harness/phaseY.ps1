# Phase Y: localize the llama.cpp Q4_K_M + LoRA collapse (MetaMathQA adapter on Qwen3-8B, gsm8k greedy, 100 items).
#   own F16 GGUF from the HF weights -> {Q4_K_M, Q4_K_M with f16 embeddings/output, Q4_0, Q5_K_M, Q6_K} + runtime LoRA
#   adapter merged into the HF weights first -> {Q4_K_M, Q8_0}, no runtime LoRA
. "$PSScriptRoot\common.ps1"
$ErrorActionPreference = 'Continue'
$snap = (Get-ChildItem 'C:\Users\patri\Documents\Codex\models\hf\models--Qwen--Qwen3-8B\snapshots' -Directory | Select-Object -First 1).FullName
$M = 'C:\Users\patri\Documents\Codex\models\gguf\phaseY'
New-Item -ItemType Directory -Force $M | Out-Null
$src = 'C:\Users\patri\Documents\Codex\llama.cpp-src'
$quant = 'C:\Users\patri\Documents\Codex\BitAgent-gym\llama-prism\bin\llama-quantize.exe'
$mmDir = "$AD\tianzl66__Qwen3-8B-MetaMathQA-50K-LoRA"
$mm = "$mmDir\lora-f16.gguf"
$env:PYTHONPATH = "$src\gguf-py"
$env:HF_HUB_OFFLINE = '1'
$D = "$R\results\diagnostics"
$G = @('--n', '100', '--temperature', '0', '--top-p', '1', '--top-k', '0')

function Convert([string]$hf, [string]$out) {
    if (Test-Path $out) { return }
    Log "phase Y: convert $hf -> $out"
    & $TorchPy -X utf8 "$src\convert_hf_to_gguf.py" $hf --outtype f16 --outfile $out *> "$out.log"
    Log ("phase Y: convert exit=$LASTEXITCODE size=" + [math]::Round((Get-Item $out -ErrorAction SilentlyContinue).Length / 1GB, 2) + " GB")
}
function Quant([string]$in, [string]$out, [string]$type, [string[]]$extra = @()) {
    if (Test-Path $out) { return }
    & $quant @extra $in $out $type 8 *> "$out.log"
    Log ("phase Y: quantize $type $(Split-Path $out -Leaf) exit=$LASTEXITCODE size=" + [math]::Round((Get-Item $out -ErrorAction SilentlyContinue).Length / 1GB, 2) + " GB")
}
function Eval([string]$name, [string]$model, [string[]]$lora) {
    if (-not (Test-Path $model)) { Log "phase Y: missing $model"; return }
    WaitGpu 11000
    & "$R\harness\serve.ps1" -Name "y-$name" -Model $model -Lora $lora -Port 8096 -Np 4 -Ctx 32768 | ForEach-Object { Log $_ }
    Gen 'gsm8k' "diag:y-${name}:greedy" "$D\gsm8k\y-$name-greedy" $G
    StopServer 8096
}

# 1) own F16 + quant variants
$F16 = "$M\Qwen3-8B-F16.gguf"
Convert $snap $F16
Quant $F16 "$M\Q4_K_M.gguf" 'Q4_K_M'
Quant $F16 "$M\Q4_K_M-f16eo.gguf" 'Q4_K_M' @('--token-embedding-type', 'f16', '--output-tensor-type', 'f16')
Quant $F16 "$M\Q4_0.gguf" 'Q4_0'
Quant $F16 "$M\Q5_K_M.gguf" 'Q5_K_M'
Quant $F16 "$M\Q6_K.gguf" 'Q6_K'

# 2) merge first, then quantize (no runtime LoRA)
$mergedHf = "$M\merged-metamath-hf"
if (-not (Test-Path "$M\merged-Q4_K_M.gguf")) {
    if (-not (Test-Path "$mergedHf\model.safetensors.index.json")) {
        Log 'phase Y: merge MetaMathQA into the HF weights'
        & $TorchPy -X utf8 "$R\harness\merge_lora.py" --base $snap --adapter $mmDir --out $mergedHf *> "$M\merge.log"
        Log ("phase Y: merge exit=$LASTEXITCODE " + ((Get-Content "$M\merge.log" | Select-String 'merged|adapter:' | ForEach-Object { $_.Line.Trim() }) -join ' | '))
    }
    $mF16 = "$M\merged-F16.gguf"
    Convert $mergedHf $mF16
    Quant $mF16 "$M\merged-Q4_K_M.gguf" 'Q4_K_M'
    Quant $mF16 "$M\merged-Q8_0.gguf" 'Q8_0'
    if ((Test-Path "$M\merged-Q4_K_M.gguf") -and (Test-Path "$M\merged-Q8_0.gguf")) {
        Remove-Item $mF16 -Force; Remove-Item $mergedHf -Recurse -Force   # 33 GB of intermediates
    }
}

# 3) evaluate (greedy, 100 gsm8k items; same items as phase X)
Eval 'own-q4km-base' "$M\Q4_K_M.gguf" @()
Eval 'own-q4km-metamath' "$M\Q4_K_M.gguf" @($mm)
Eval 'q4km-f16eo-metamath' "$M\Q4_K_M-f16eo.gguf" @($mm)
Eval 'q4_0-metamath' "$M\Q4_0.gguf" @($mm)
Eval 'q5km-metamath' "$M\Q5_K_M.gguf" @($mm)
Eval 'q6k-metamath' "$M\Q6_K.gguf" @($mm)
Eval 'merged-q4km' "$M\merged-Q4_K_M.gguf" @()
Eval 'merged-q8' "$M\merged-Q8_0.gguf" @()
Log 'phase Y: done'
