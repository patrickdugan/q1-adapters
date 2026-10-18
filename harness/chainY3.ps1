# After phase Y pass 2: is the runtime-LoRA failure on Q4_K_M CUDA-specific? Same official Q4_K_M + MetaMathQA,
# CPU backend only: -ngl 0 AND the CUDA device hidden from the server process (otherwise llama.cpp may still
# offload large prompt batches to the GPU), greedy, first 30 gsm8k items.
. "$PSScriptRoot\common.ps1"
$mark = (Get-Content (Join-Path $R 'runs\pipeline.log') | Measure-Object -Line).Lines
while ($true) {
    $tail = Get-Content (Join-Path $R 'runs\pipeline.log') | Select-Object -Skip $mark
    if ($tail -match 'phase Y: done') { break }
    Start-Sleep -Seconds 60
}
$mm = $Online['gsm8k-metamath']
$D = "$R\results\diagnostics"
$G = @('--n', '30', '--temperature', '0', '--top-p', '1', '--top-k', '0')
$env:CUDA_VISIBLE_DEVICES = '-1'   # inherited by the server started below
& "$R\harness\serve.ps1" -Name 'y-cpu-q4km-metamath' -Model $Q4 -Lora @($mm) -Port 8096 -Np 4 -Ctx 16384 -Ngl 0 | ForEach-Object { Log $_ }
Remove-Item Env:\CUDA_VISIBLE_DEVICES
Gen 'gsm8k' 'diag:y-cpu-q4km-metamath:greedy' "$D\gsm8k\y-cpu-q4km-metamath-greedy" $G
StopServer 8096
Log 'phase Y3 (cpu): done'
