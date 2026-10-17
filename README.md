# q1-adapters

Task LoRAs on a natively 1-bit 8B model (PrismML **Bonsai 8B**, `Q1_0`, 1.16 GB), trained from
**Prime Intellect Environments Hub** rewards on one laptop GPU, benchmarked against a 4-bit
**Qwen3-8B** (`Q4_K_M`, 5.03 GB) carrying community adapters found on the Hugging Face Hub.

Paper draft: [`paper/q1_adapters.md`](paper/q1_adapters.md). Tables: [`results/tables.md`](results/tables.md),
routing: [`results/routing.md`](results/routing.md).

Headline (200 items per env; GSM8K/MedMCQA also at 500):

| | Q4 Qwen3-1.7B (1.11 GB) | Q1 Bonsai 8B (1.16 GB) | Q1 + routed adapters (+4×87 MB) | Q4 Qwen3-8B (5.03 GB) |
|---|---|---|---|---|
| reverse-text | 0.079 | 0.115 | **0.817** | 0.178 |
| gsm8k | 0.800 | 0.845 | 0.875 | **0.945** |
| medmcqa | 0.425 | 0.375 | 0.440 | **0.515** |
| ifeval | 0.590 | 0.640 | 0.640 | **0.680** |
| macro | 0.474 | 0.494 | **0.693** | 0.580 |
| GPU decode | 320 tok/s | 201 tok/s | — | 89 tok/s |

Also: a 72k-parameter TRM router dispatches the adapters at oracle level on full prompts; lexical LDT
lattices hurt as hard filters; and public Qwen3-8B LoRAs collapse on llama.cpp `Q4_K_M` GGUFs while
working on `Q8_0` and on NF4 + PEFT.

## Environments (scored by the env packages' own reward code)

| Hub env | task | score reported |
|---|---|---|
| `primeintellect/reverse-text` | reverse a sentence character by character | LCS ratio of the parsed `<reversed_text>` |
| `primeintellect/gsm8k` | grade-school math | math-verify on `#### N` |
| `maziyar/OpenMed_MedMCQA` | medical multiple choice | boxed-letter accuracy (env reward also logged) |
| `arcee-ai/ifeval` | verifiable instruction constraints | RLVR-IFEval pass rate |

## Layout

- `harness/envs.py` — loads each env from its installed package (dataset, prompt, reward funcs).
- `harness/generate.py` — runs an env against an OpenAI-compatible server (llama-server), scores, writes JSONL + summary.
- `harness/export_gold.py`, `harness/train_lora.py` — training data (env gold / rejection sampling) and QLoRA on Bonsai, exported to GGUF.
- `harness/serve.ps1`, `harness/phase{A,B,C,D}.ps1` — server launcher and the four experiment phases.
- `harness/make_tables.py` — tables, paired bootstrap deltas, score-vs-memory figure.
- `results/<env>/<config>.jsonl` — every sample with its score; `*.summary.json` per run.

## Environments used to run it (Windows, RTX 5080 Laptop 16 GB)

- `vf` venv: `verifiers<0.4` (the classic-API envs do not load on 0.4), the four env packages from the Hub index, `langdetect`.
- torch venv: torch 2.9 + cu128, transformers 4.57, peft 0.21, bitsandbytes 0.50.
- PrismML llama.cpp fork (build 10743) for every model (Q1_0 requires it); upstream `convert_lora_to_gguf.py`.

Weights (Bonsai, Qwen3 GGUF, all adapters) live outside the repo and are not committed.
