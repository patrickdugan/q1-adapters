# q1-adapters

Task LoRAs on a natively 1-bit 8B model (PrismML **Bonsai 8B**, `Q1_0`, 1.16 GB), trained from
**Prime Intellect Environments Hub** rewards on one laptop GPU, benchmarked against a 4-bit
**Qwen3-8B** (`Q4_K_M`, 5.03 GB) carrying community adapters found on the Hugging Face Hub.

Paper draft: [PDF](paper/q1_adapters.pdf) · [Markdown](paper/q1_adapters.md). Tables:
[`results/tables.md`](results/tables.md), routing: [`results/routing.md`](results/routing.md).
Adapters (PEFT + GGUF): [huggingface.co/AlephFunk/bonsai-8b-prime-env-adapters](https://huggingface.co/AlephFunk/bonsai-8b-prime-env-adapters).

Headline (200 items per env; GSM8K/MedMCQA also at 500):

| | Q4 Qwen3-1.7B (1.11 GB) | Q1 Bonsai 8B (1.16 GB) | Q1 + routed adapters (+4×87 MB) | Q4 Qwen3-8B (5.03 GB) |
|---|---|---|---|---|
| reverse-text | 0.079 | 0.115 | **0.817** | 0.208 |
| gsm8k | 0.800 | 0.845 | 0.875 | **0.940** |
| medmcqa | 0.425 | 0.375 | 0.440 | **0.560** |
| ifeval | 0.590 | 0.640 | 0.640 | **0.685** |
| macro | 0.474 | 0.494 | **0.693** | 0.598 |
| GPU decode | 320 tok/s | 201 tok/s | — | 89 tok/s |

More environments, bigger samples (§3.8; non-thinking mode; ordered by reasoning load):

| | n | Q4 Qwen3-1.7B | Q1 Bonsai 8B | Q1 + our adapter | Q4 Qwen3-8B |
|---|---|---|---|---|---|
| gsm8k (full test) | 1,319 | 0.798 | 0.851 | 0.871 | **0.933** |
| medmcqa | 1,000 | 0.437 | 0.409 | 0.453 | **0.567** |
| pubmedqa | 500 | 0.390 | 0.364 | **0.434** | 0.432 |
| ascii-tree | 200 | 0.106 | 0.062 | **0.981** | 0.230 |
| math500 | 500 | 0.710 | 0.642 | 0.630 (GSM8K adapter) | **0.810** |
| reasoning-core | 500 | 0.175 | 0.220 | **0.300** | 0.256 |
| aime 24+25 (avg@2) | 60 | 0.117 | 0.067 | — | **0.242** |

The reasoning-heavy envs are weak for every 8B-class model in non-thinking mode, and on multi-step
math the 1-bit 8B falls behind even the memory-matched 4-bit 1.7B. Adapters help most where the
answer is a transformation of the prompt (reverse-text, ascii-tree), not where it must be derived.

Also:
- a 72k-parameter TRM router dispatches the adapters at oracle level on full prompts; lexical LDT
  lattices hurt as hard filters;
- Qwen's official `Qwen3-8B-Q4_K_M.gguf` is built from an **AWQ-rescaled checkpoint**
  (`general.name` "Qwen3 8B Awq Compatible Instruct"; norm weights differ 20–55%), so public LoRAs
  trained on `Qwen/Qwen3-8B` are misapplied on it (MetaMathQA GSM8K 0.98 → 0.37) — on a `Q4_K_M`
  converted from the real checkpoint they work (0.86); the 4-bit numbers above use that file;
- even correctly applied, none of the four public adapters improved a Hub env's score (format
  mismatch, a stop-token defect, plain trade-offs); the env-trained adapters did.

## Environments (scored by the env packages' own reward code)

| Hub env | task | score reported |
|---|---|---|
| `primeintellect/reverse-text` | reverse a sentence character by character | LCS ratio of the parsed `<reversed_text>` |
| `primeintellect/gsm8k` | grade-school math | math-verify on `#### N` |
| `maziyar/OpenMed_MedMCQA` | medical multiple choice | boxed-letter accuracy (env reward also logged) |
| `arcee-ai/ifeval` | verifiable instruction constraints | RLVR-IFEval pass rate |
| `maziyar/OpenMed_PubMedQA` | yes / no / maybe from a PubMed abstract | answer accuracy (env reward also logged) |
| `primeintellect/ascii-tree` | prose directory description → ASCII tree | env's weighted tree similarity |
| `primeintellect/math500` | MATH-500 competition math | math-verify on the last `\boxed{}` |
| `primeintellect/aime2024` + `aime2025` | AIME problems (60) | math-verify on the last `\boxed{}` |
| `primeintellect/reasoning-core` | 56 procedurally generated symbolic task families | env's per-task scorer |

## Layout

- `harness/envs.py` — loads each env from its installed package (dataset, prompt, reward funcs).
- `harness/generate.py` — runs an env against an OpenAI-compatible server (llama-server), scores, writes JSONL + summary.
- `harness/export_gold.py`, `harness/train_lora.py` — training data (env gold / rejection sampling) and QLoRA on Bonsai, exported to GGUF.
- `harness/serve.ps1`, `harness/phase*.ps1` — server launcher and the experiment phases (`phaseN.ps1` /
  `phaseN2.ps1` = §3.8). Phases resume per item; `harness/resume.ps1` (with `install_resume_task.ps1`,
  a per-user logon task) relaunches a phase that a crash or forced restart interrupted.
- `harness/make_tables.py` — tables, paired bootstrap deltas, score-vs-memory figure.
- `results/<env>/<config>.jsonl` — every sample with its score; `*.summary.json` per run.

## Environments used to run it (Windows, RTX 5080 Laptop 16 GB)

- `vf` venv: `verifiers<0.4` (the classic-API envs do not load on 0.4), the nine env packages from the Hub index, `langdetect`.
- torch venv: torch 2.9 + cu128, transformers 4.57, peft 0.21, bitsandbytes 0.50.
- PrismML llama.cpp fork (build 10743) for every model (Q1_0 requires it); upstream `convert_lora_to_gguf.py`.

Weights (Bonsai, Qwen3 GGUF, all adapters) live outside the repo and are not committed.
