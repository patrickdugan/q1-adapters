| env (metric) | n | Q4 1.7B base (1.11 GB) | Q1 8B base (1.16 GB) | Q1 + our adapter (1.25 GB) | Q1 + online Qwen3 adapter | Q4 8B base (5.03 GB) | Q4 8B + online adapter (5.11 GB) | Q4 8B + our Bonsai adapter |
|---|---|---|---|---|---|---|---|---|
| reverse-text (LCS ratio) | 200 | — | 0.115 | 0.817 | — | — | — | — |
| gsm8k (accuracy) | 200 | — | 0.845 | 0.875 | 0.870 (gsm8k-grpo) | — | — | — |
| medmcqa (accuracy) | 200 | — | 0.375 | 0.440 | 0.420 (medmcqa-vireshk) | — | — | — |
| ifeval (constraint pass rate) | 200 | — | 0.640 | 0.640 | 0.630 (ifeval-tulu) | — | — | — |

Paired deltas (same examples, same seeds; bootstrap 95% CI):
- reverse-text q1 ours vs q1 base: +0.702 [+0.682, +0.722] (n=200)
- gsm8k q1 online vs q1 base: +0.025 [-0.025, +0.075] (n=200)
- gsm8k q1 ours vs q1 base: +0.030 [-0.020, +0.075] (n=200)
- medmcqa q1 online vs q1 base: +0.045 [-0.030, +0.120] (n=200)
- medmcqa q1 ours vs q1 base: +0.065 [-0.005, +0.135] (n=200)
- ifeval q1 online vs q1 base: -0.010 [-0.050, +0.025] (n=200)
- ifeval q1 ours vs q1 base: +0.000 [-0.045, +0.040] (n=200)

Served throughput during the eval runs (6 concurrent requests, aggregate decode tokens/s):

| env | configuration | tokens/s |
|---|---|---|
| reverse-text | q1+ours | 119 |
| reverse-text | q1-base | 398 |
| gsm8k | q1+online:gsm8k-grpo | 161 |
| gsm8k | q1+online:gsm8k-metamath | 186 |
| gsm8k | q1+ours | 184 |
| gsm8k | q1-base | 389 |
| medmcqa | q1+online:medmcqa-vireshk | 175 |
| medmcqa | q1+ours | 183 |
| medmcqa | q1-base | 389 |
| ifeval | q1+ours | 166 |
