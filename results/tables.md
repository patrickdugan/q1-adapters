| env (metric) | n | Q4 1.7B base (1.11 GB) | Q1 8B base (1.16 GB) | Q1 + our adapter (1.25 GB) | Q1 + online Qwen3 adapter | Q4 8B base (5.03 GB) | Q4 8B + online adapter (5.11 GB) | Q4 8B + our Bonsai adapter |
|---|---|---|---|---|---|---|---|---|
| reverse-text (LCS ratio) | 200 | 0.079 | 0.115 | 0.817 | — | 0.208 | — | 0.282 |
| gsm8k (accuracy) | 200 | 0.800 | 0.845 | 0.875 | 0.870 (gsm8k-grpo) | 0.940 | 0.860 (gsm8k-metamath) | 0.940 |
| medmcqa (accuracy) | 200 | 0.425 | 0.375 | 0.440 | 0.420 (medmcqa-vireshk) | 0.560 | 0.085 (medmcqa-vireshk) | 0.590 |
| ifeval (constraint pass rate) | 200 | 0.590 | 0.640 | 0.640 | 0.630 (ifeval-tulu) | 0.685 | 0.540 (ifeval-tulu) | 0.690 |

Q4 8B columns above: our own Q4_K_M from the exact Qwen/Qwen3-8B weights. Qwen's official Qwen3-8B-Q4_K_M.gguf (general.name 'Qwen3 8B Awq Compatible Instruct'; AWQ-rescaled norms) for comparison:

| env | official Q4_K_M base | official + best online adapter | official + our Bonsai adapter |
|---|---|---|---|
| reverse-text | 0.178 | — | 0.093 |
| gsm8k | 0.938 | 0.425 (gsm8k-metamath) | 0.940 |
| medmcqa | 0.518 | 0.475 (medmcqa-vireshk) | 0.510 |
| ifeval | 0.680 | 0.375 (ifeval-tulu) | 0.660 |

Paired deltas (same examples, same seeds; bootstrap 95% CI):
- reverse-text q1 base vs q4 1.7b base: +0.037 [+0.015, +0.059] (n=200)
- reverse-text q1 ours vs q1 base: +0.702 [+0.682, +0.722] (n=200)
- reverse-text q1 ours vs q4 base: +0.609 [+0.577, +0.644] (n=200)
- gsm8k q1 base vs q4 1.7b base: +0.045 [-0.015, +0.105] (n=200)
- gsm8k q1 online vs q1 base: +0.025 [-0.025, +0.075] (n=200)
- gsm8k q1 ours vs q1 base: +0.030 [-0.020, +0.075] (n=200)
- gsm8k q4 online vs q4 base: -0.080 [-0.130, -0.030] (n=200)
- gsm8k q1 ours vs q4 base: -0.065 [-0.110, -0.020] (n=200)
- gsm8k q1 ours vs q4 online: +0.015 [-0.035, +0.070] (n=200)
- medmcqa q1 base vs q4 1.7b base: -0.050 [-0.130, +0.035] (n=200)
- medmcqa q1 online vs q1 base: +0.045 [-0.030, +0.120] (n=200)
- medmcqa q1 ours vs q1 base: +0.065 [-0.005, +0.135] (n=200)
- medmcqa q4 online vs q4 base: -0.475 [-0.550, -0.400] (n=200)
- medmcqa q1 ours vs q4 base: -0.120 [-0.200, -0.030] (n=200)
- medmcqa q1 ours vs q4 online: +0.355 [+0.275, +0.440] (n=200)
- ifeval q1 base vs q4 1.7b base: +0.050 [-0.005, +0.105] (n=200)
- ifeval q1 online vs q1 base: -0.010 [-0.050, +0.025] (n=200)
- ifeval q1 ours vs q1 base: +0.000 [-0.045, +0.040] (n=200)
- ifeval q4 online vs q4 base: -0.145 [-0.200, -0.090] (n=200)
- ifeval q1 ours vs q4 base: -0.045 [-0.085, +0.000] (n=200)
- ifeval q1 ours vs q4 online: +0.100 [+0.045, +0.155] (n=200)

Extended evaluation, 500 examples (configurations run to 500; each row averaged over the ids all of its present configurations share):

| env | n | Q4 1.7B base | Q1 base | Q1 + our adapter | Q1 + GRPO-GSM8K | Q1 + MetaMathQA | Q1 + MedMCQA LoRA | Q4 8B base |
|---|---|---|---|---|---|---|---|---|
| gsm8k | 500 | 0.802 | 0.854 | 0.868 | 0.860 | 0.842 | — | 0.938 |
- gsm8k (n=500) Q1 + our adapter vs Q1 base: +0.014 [-0.014, +0.044]
- gsm8k (n=500) Q1 base vs Q4 1.7B base: +0.052 [+0.014, +0.088]
- gsm8k (n=500) Q1 + our adapter vs Q4 1.7B base: +0.066 [+0.032, +0.102]
- gsm8k (n=500) Q1 + our adapter vs Q4 8B base: -0.070 [-0.096, -0.046]
| medmcqa | 500 | 0.392 | 0.388 | 0.450 | — | — | 0.408 | 0.518 |
- medmcqa (n=500) Q1 + our adapter vs Q1 base: +0.062 [+0.018, +0.108]
- medmcqa (n=500) Q1 base vs Q4 1.7B base: -0.004 [-0.048, +0.044]
- medmcqa (n=500) Q1 + our adapter vs Q4 1.7B base: +0.058 [+0.010, +0.106]
- medmcqa (n=500) Q1 + our adapter vs Q4 8B base: -0.068 [-0.122, -0.014]

llama-bench (PrismML fork build 10743; GPU = RTX 5080 Laptop, -ngl 99; CPU = 8 threads):

| model | device | test | tokens/s | model size (GB) |
|---|---|---|---|---|
| q1 | gpu | pp512 | 4948.8 | 1.15 |
| q1 | gpu | tg128 | 200.6 | 1.15 |
| q1 | cpu | pp128 | 782.4 | 1.15 |
| q1 | cpu | tg32 | 17.7 | 1.15 |
| q4-1.7b | gpu | pp512 | 16361.6 | 1.10 |
| q4-1.7b | gpu | tg128 | 319.5 | 1.10 |
| q4-1.7b | cpu | pp128 | 1142.1 | 1.10 |
| q4-1.7b | cpu | tg32 | 51.8 | 1.10 |
| q4 | gpu | pp512 | 3137.3 | 5.02 |
| q4 | gpu | tg128 | 89.1 | 5.02 |
| q4 | cpu | pp128 | 336.9 | 5.02 |
| q4 | cpu | tg32 | 11.1 | 5.02 |

Deployment footprint (llama-server, 1 slot, 4k context; VRAM = nvidia-smi delta, RAM = working set):

| server | weights file | adapters | VRAM | process RAM |
|---|---|---|---|---|
| q1-bonsai-8b-nolora | 1.16 GB | 0 MB | 1.87 GiB | 1.42 GiB |
| q1-bonsai-8b-lora4 | 1.16 GB | 349 MB | 2.37 GiB | 1.55 GiB |
| q4-qwen3-8b-nolora | 5.03 GB | 0 MB | 5.23 GiB | 4.78 GiB |
| q4-qwen3-8b-lora4 | 5.03 GB | 349 MB | 5.73 GiB | 4.91 GiB |
| q4-qwen3-1.7b-nolora | 1.11 GB | 0 MB | 1.73 GiB | 1.44 GiB |

Served throughput during the eval runs (6 concurrent requests, aggregate decode tokens/s):

| env | configuration | tokens/s |
|---|---|---|
| reverse-text | q1+ours:gsm8k | 291 |
| reverse-text | q1+ours:ifeval | 271 |
| reverse-text | q1+ours:medmcqa | 300 |
| reverse-text | q1+ours | 119 |
| reverse-text | q1-base | 398 |
| reverse-text | q4+bonsai-adapter | 64 |
| reverse-text | q4-1.7b-base | 595 |
| reverse-text | q4-base | 52 |
| reverse-text | q4own+bonsai-adapter | 108 |
| reverse-text | q4own-base | 171 |
| gsm8k | q1+online:gsm8k-grpo | 256 |
| gsm8k | q1+online:gsm8k-metamath | 257 |
| gsm8k | q1+ours:ifeval | 315 |
| gsm8k | q1+ours:medmcqa | 302 |
| gsm8k | q1+ours:reverse-text | 199 |
| gsm8k | q1+ours | 257 |
| gsm8k | q4+bonsai-adapter | 85 |
| gsm8k | q4+online:gsm8k-grpo | 120 |
| gsm8k | q4+online:gsm8k-metamath | 58 |
| gsm8k | q4-1.7b-base | 624 |
| gsm8k | q4-base | 345 |
| gsm8k | q4own+bonsai-adapter | 117 |
| gsm8k | q4own+online:gsm8k-grpo | 96 |
| gsm8k | q4own+online:gsm8k-metamath | 82 |
| gsm8k | q4own-base | 151 |
| medmcqa | q1+online:medmcqa-vireshk | 230 |
| medmcqa | q1+ours:gsm8k | 296 |
| medmcqa | q1+ours:ifeval | 152 |
| medmcqa | q1+ours:reverse-text | 301 |
| medmcqa | q1+ours | 251 |
| medmcqa | q4+bonsai-adapter | 101 |
| medmcqa | q4+online:medmcqa-vireshk | 71 |
| medmcqa | q4-1.7b-base | 622 |
| medmcqa | q4-base | 538 |
| medmcqa | q4own+bonsai-adapter | 105 |
| medmcqa | q4own+online:medmcqa-vireshk | 94 |
| medmcqa | q4own-base | 148 |
| ifeval | q1+ours:gsm8k | 270 |
| ifeval | q1+ours:medmcqa | 287 |
| ifeval | q1+ours:reverse-text | 244 |
| ifeval | q1+ours | 166 |
| ifeval | q4+bonsai-adapter | 110 |
| ifeval | q4+online:ifeval-tulu | 48 |
| ifeval | q4-1.7b-base | 551 |
| ifeval | q4-base | 63 |
| ifeval | q4own+bonsai-adapter | 83 |
| ifeval | q4own+online:ifeval-tulu | 157 |
| ifeval | q4own-base | 131 |
