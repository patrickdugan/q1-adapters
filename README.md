# q1-adapters

Task LoRAs on a natively 1-bit 8B model (PrismML **Bonsai 8B**, `Q1_0`, 1.16 GB), trained from
**Prime Intellect Environments Hub** rewards on one laptop GPU, benchmarked against a 4-bit
**Qwen3-8B** (`Q4_K_M`, 5.03 GB) carrying community adapters found on the Hugging Face Hub.

Paper draft: [`paper/q1_adapters.md`](paper/q1_adapters.md). Tables: [`results/tables.md`](results/tables.md).

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
