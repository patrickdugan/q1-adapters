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

**Routers.** A skill library needs a router. Following the TRM/LDT split of Dugan (2026,
*Trade-offs between TRM/LDT hybrids*), we compare a **TRM** — a tiny recursive classifier
(72k parameters: hashed word/bigram/char-trigram features, a latent *z* and answer state *y*
refined by two weight-shared 2-layer MLPs, 3 inner × 3 outer steps, deep supervision) — with an
**LDT** lexical lattice (each known word carries the candidate set of envs it occurs in; a
prompt's state is the meet over its words; the empty set is bottom), and the paper's hybrid control
policies: soft (lattice as telemetry), hard filter, confidence arbitration, and a typed membrane
that **abstains to the base model** when the TRM's proposal conflicts with the lattice. Routers are
trained on 800 train prompts per env and scored on the 800 eval prompts in two views: the full
prompt as the env sends it, and the user's content with the env's instruction scaffolding removed.
Besides routing accuracy we report **end-to-end** score: the env score of the answer the 1-bit model
actually produced under the chosen adapter (all adapter × env answers are generated, so this is a
lookup, not a model of the outcome).

## 3. Results

<!-- MAIN TABLE -->

### 3.2 What an 87 MB adapter buys on the 1-bit base

Paired deltas over the same 200 examples and seeds (bootstrap 95% CI):

| env | Q1 base | Q1 + our adapter | Δ | what the adapter learned |
|---|---|---|---|---|
| reverse-text | 0.115 | **0.817** | **+0.702** [+0.682, +0.722] | the task itself: the base reorders words instead of reversing characters and runs past the 384-token cap on half the items (mean 205 tokens, 49% truncated); with the adapter, 51 tokens, 0% truncated |
| gsm8k | 0.845 | 0.875 | +0.030 [−0.020, +0.075] | sharpening only — self-distillation on its own correct answers (loss 0.12 → 0.08) |
| medmcqa | 0.375 | 0.440 | +0.065 [−0.005, +0.135] | modest; borderline at n = 200 (500-example extension in §3.6) |
| ifeval | 0.640 | 0.640 | 0.000 [−0.045, +0.040] | nothing measurable |

The pattern is what one would predict for rejection-sampling fine-tuning: it can only amplify
behaviour the base already samples. Where the base is far from the task (reverse-text) a few
hundred gold examples move it by 0.7; where it already samples correct answers most of the time
(gsm8k 91% of training samples correct) there is little left to amplify.

**Training cost** (RTX 5080 Laptop, 16 GB; the QLoRA job sat at the VRAM limit, so these are
upper bounds):

| adapter | data | sampling (Q1 server) | training | total |
|---|---|---|---|---|
| reverse-text | 800 env gold reversals | — | 7 min (150 steps) | 7 min |
| gsm8k | 757 of 1,600 samples correct | 17 min | 50 min (96 steps) | 67 min |
| medmcqa | 465 of 1,600 | 16 min | 60 min (60 steps) | 76 min |
| ifeval | 479 of 1,600 | 23 min | 80 min (60 steps) | 103 min |

### 3.3 Public Qwen3-8B adapters run on the 1-bit model

Bonsai shares Qwen3-8B's architecture, so rank-16 LoRAs trained on Qwen3-8B load onto the 1-bit
weights unchanged. They change every output (0/200 texts identical to base on gsm8k and medmcqa,
33/200 on ifeval) but their effect is within noise at n = 200: gsm8k GRPO adapter +0.025
[−0.025, +0.075], MetaMathQA −0.005, MedMCQA adapter +0.045 [−0.030, +0.120], Tulu IF adapter
−0.010 [−0.050, +0.025]. They neither break the 1-bit model nor reliably help it.

### 3.4 Adapters are narrow — and one is dangerous off-task

Every adapter on every env (1-bit base, 200 examples per cell; rows = env, columns = adapter):

| env | no adapter | reverse-text | gsm8k | medmcqa | ifeval |
|---|---|---|---|---|---|
| reverse-text | 0.115 | **0.817** | 0.131 | 0.133 | 0.135 |
| gsm8k | 0.845 | **0.045** | **0.875** | 0.850 | 0.875 |
| medmcqa | 0.375 | **0.250** | 0.370 | **0.440** | 0.340 |
| ifeval | 0.640 | **0.330** | 0.645 | 0.630 | **0.640** |

The three self-distilled adapters are close to harmless off-task. The reverse-text adapter — the
one that learned a genuinely new behaviour — destroys the others (gsm8k 0.845 → 0.045: it starts
reversing). A skill library on one base is therefore only as good as its router.

### 3.5 Routing the library: TRM, LDT and hybrids

End-to-end macro score (mean over the four envs of the score of the answer actually produced under
the routed adapter; 800 prompts, TRM averaged over 5 seeds):

| router | full prompt: routing acc | full: end-to-end | content only: routing acc | content: end-to-end |
|---|---|---|---|---|
| oracle (true env's adapter) | 1.000 | 0.693 | 1.000 | 0.693 |
| no adapter (base) | — | 0.494 | — | 0.494 |
| best single adapter for everything (medmcqa) | — | 0.513 | — | 0.513 |
| **TRM** (72k parameters) | 0.998 | **0.693** | 0.865 | **0.615** |
| LDT alone (lexical lattice; bottom → base) | 0.305 | 0.506 | 0.305 | 0.507 |
| hard-filter hybrid | 0.935 | 0.693 | 0.683 | 0.553 |
| soft hybrid / confidence arbitration / membrane (γ by CV) | = TRM | 0.693 | = TRM | 0.615 |
| membrane, abstain threshold tuned end-to-end (cross-fitted) | 0.998 | 0.693 | 0.862 | 0.613 |
| TRM + expected-utility over the cross-task matrix (cross-fitted) | 0.643 | 0.684 | 0.570 | 0.603 |

Findings, which replicate the routing ablation of the TRM/LDT hybrid paper on a new problem:

1. **A 72k-parameter TRM is enough when the prompt carries the task**: it matches the oracle
   (0.693), turning four 87 MB adapters into +0.20 macro score over the base model, while no single
   adapter gets more than +0.02.
2. **Lexical lattice evidence is too noisy to filter with.** The LDT alone reaches bottom on most
   prompts (abstains 43–63%), and using it as a hard filter costs 0.06 end-to-end on content-only
   prompts — the same failure mode as the paper's routing benchmark (0.584 vs 0.815). As soft
   telemetry it is harmless and inert.
3. **Abstention does not rescue content-only routing here.** The errors are bare sentences
   (reverse-text items) routed to other adapters, and for those the base model is no better than a
   wrong adapter (0.115), so abstaining gains nothing; the costly error (a math problem sent to
   the reverse-text adapter) is rare (8/200). Expected-utility routing over the measured cross-task
   matrix is *worse* (0.603): off-task, three adapters are near-ties, and the decision chases noise
   in the estimated matrix.

<!-- Q4 COMPARISON -->

<!-- SYSTEMS -->
