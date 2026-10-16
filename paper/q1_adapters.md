# One gigabyte, many skills: task adapters on a 1-bit 8B model, scored on Prime Intellect environments

*Draft — results sections are filled from `results/` by the run that produced them.*

## Abstract

<!-- ABSTRACT -->

## 1. Question

A 4-bit 8B model is the usual "runs on a laptop" configuration: about 5 GB of weights. PrismML's
**Bonsai 8B** has end-to-end 1-bit weights — embeddings, attention, MLP and LM head — on the
Qwen3-8B dense architecture: `Q1_0` stores one sign bit per weight plus one FP16 scale per group of
128 (1.125 bits/weight, 1.16 GB). PrismML reports GSM8K 88 and IFEval 79.8 for it (EvalScope, vLLM,
thinking settings per their card; Apache-2.0). A 1-bit model is weaker out of the box, so the question is not "is Q1 as good as Q4"
but **what Q1 buys once you add the cheapest specialization available — a small task LoRA — and how
that compares with the 4-bit model plus the adapters people already publish.**

We answer it on four public **Prime Intellect Environments Hub** environments, using each
environment's own dataset, prompt and reward code, so every number below is the score the Hub
environment would assign.

## 2. Setup

**Models.**
- *Q1*: `prism-ml/Bonsai-8B-gguf`, `Bonsai-8B-Q1_0.gguf` (1,158,654,496 bytes), served by PrismML's
  llama.cpp fork (build 10743).
- *Q4*: `Qwen/Qwen3-8B-GGUF`, `Qwen3-8B-Q4_K_M.gguf` (5,027,783,488 bytes), served by the same binary.

Both use the same Qwen3 chat template (taken from the Bonsai checkpoint), which always closes an
empty `<think>` block — i.e. non-thinking mode for every configuration — so prompts are
byte-identical across models.

**Environments** (installed from the Hub index; scored by their own reward functions):

| Hub env | eval set | score |
|---|---|---|
| `primeintellect/reverse-text` | last 200 of the env's 1,000 rows (first 800 = train) | LCS ratio of parsed `<reversed_text>` |
| `primeintellect/gsm8k` | first 200 of GSM8K test | math-verify on `#### N` |
| `maziyar/OpenMed_MedMCQA` | first 200 of the env's validation split | boxed-letter accuracy |
| `arcee-ai/ifeval` | the env's own seeded 200-row holdout | RLVR-IFEval constraint pass rate |

Implementation notes: the three classic-API envs require `verifiers<0.4` (0.4.0 removed
`vf.Environment`); `primeintellect/gsm8k` is a verifiers-v1 taskset whose `verify.py` we run
in-process with math-verify's signal/process timeouts disabled (on Windows they spawn a child that
cannot re-run the script and every answer scores 0) — the scoring logic is otherwise line-for-line.

**Sampling.** Qwen3's recommended non-thinking settings (T = 0.7, top-p 0.8, top-k 20), one sample
per eval example with a per-example seed shared by all configurations, so comparisons are paired.
Max new tokens: 384 (reverse-text), 768 (gsm8k), 1024 (medmcqa, ifeval).

**Adapters.**
- *Ours (Bonsai)*: rank-16 LoRA on all seven projections (43.6 M parameters; 87 MB as an f16 GGUF),
  trained with QLoRA on `prism-ml/Bonsai-8B-unpacked` in 4-bit NF4 — whose weights are an exact
  expansion of the 1-bit release, so the adapter applies unchanged to `Q1_0` (we verified
  byte-identical generations between the torch 4-bit path and the 1-bit server in prior work).
  Training signal comes from the environment itself: for reverse-text the env's gold reversal
  (Prime Intellect's own SFT warm-up shape); for gsm8k, medmcqa and ifeval, **rejection-sampling
  fine-tuning** — sample 4 completions per prompt for 400 train prompts from the base 1-bit model,
  keep up to two distinct completions the env scores as fully correct, fine-tune on them.
- *Online (Qwen3-8B)*: the best-matching public LoRAs on the Hub, used as published (not trained by us):
  `Pradheep1647/qwen3_8b_gsm8k_unloth_trl` (GRPO on GSM8K) and `tianzl66/Qwen3-8B-MetaMathQA-50K-LoRA`
  for gsm8k; `vireshk/qwen3-8b-medmcqa-qlora` for MedMCQA; `tianzl66/Qwen3-8B-InstructionFollowing-LoRA`
  (Tulu-3 IF split) for IFEval. All are rank 16 on all seven projections — the same shape as ours.
  No public adapter exists for reverse-text. (The most-downloaded "GSM8K LoRA",
  `swagger00/Qwen3-8B-GRPO-GSM8K`, ships merged 16 GB weights, not an adapter, and was excluded.)

**Serving.** One llama-server per base model with every adapter loaded at start
(`--lora-init-without-apply`) and switched per run through `/lora-adapters` — the deployment
pattern where one small base serves many skills.

<!-- RESULTS -->
