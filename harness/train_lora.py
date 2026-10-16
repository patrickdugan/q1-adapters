"""Train a task LoRA for Bonsai 8B (QLoRA on the unpacked weights) and export it as a llama.cpp GGUF LoRA.

  python harness/train_lora.py --env gsm8k --data runs/rft/gsm8k-q1-base.jsonl --min-score 1.0 \
         --out C:/Users/patri/Documents/Codex/models/q1-adapters/gsm8k

Data rows are generate.py / export_gold.py JSONL (messages, text, score, reward). Rejection-
sampling fine-tuning: per prompt keep the best sample whose env score >= --min-score (ties:
higher env reward, then shorter). The training string is exactly what the server sends --
Bonsai's chat template with add_generation_prompt (which already closes an empty <think>
block) -- followed by the completion and <|im_end|>; loss is on the completion only.

The base is loaded in 4-bit NF4. Bonsai's unpacked checkpoint is an exact expansion of the
1-bit release, so an adapter trained here applies unchanged to Bonsai-8B-Q1_0.gguf (verified
byte-identical generation in stego-bench). Runs in the BitAgent-gym torch venv.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,  # noqa: E402
                          TrainerCallback, TrainingArguments)

REPO_ID = "prism-ml/Bonsai-8B-unpacked"
REVISION = "376f381570d6115bc03f82adcfa4af0c7672ae54"
CACHE_DIR = r"C:\Users\patri\Documents\Codex\BitAgent-gym\runs\hf-cache"
SNAPSHOT = rf"{CACHE_DIR}\models--prism-ml--Bonsai-8B-unpacked\snapshots\{REVISION}"
LLAMA_SRC = r"C:\Users\patri\Documents\Codex\llama.cpp-src"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def select(rows: list[dict], min_score: float, per_prompt: int) -> list[dict]:
    by_id = defaultdict(list)
    for r in rows:
        if r["score"] >= min_score and (r.get("finish_reason") or "stop") != "length" and r["text"].strip():
            by_id[r["id"]].append(r)
    out = []
    for rid in sorted(by_id):
        cands = sorted(by_id[rid], key=lambda r: (-r["reward"], len(r["text"])))
        seen = set()
        for r in cands:
            if r["text"] in seen:
                continue
            seen.add(r["text"])
            out.append(r)
            if len(seen) >= per_prompt:
                break
    return out


class Rows(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(pad_id):
    def fn(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
        lab = torch.full((len(batch), n), -100, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"])
            lab[i, :L] = torch.tensor(b["labels"])
            att[i, :L] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}
    return fn


class LossLog(TrainerCallback):
    def __init__(self):
        self.hist = []

    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "loss" in logs:
            self.hist.append({"step": state.global_step, "loss": round(float(logs["loss"]), 4)})


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--data", required=True, nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--min-score", type=float, default=1.0)
    p.add_argument("--per-prompt", type=int, default=1)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--log", default=None, help="write a training record JSON here")
    args = p.parse_args(argv)

    rows = [json.loads(line) for f in args.data for line in Path(f).read_text(encoding="utf-8").split("\n")
            if line.strip()]  # not splitlines(): raw U+2028 inside JSON strings would split a record
    chosen = select(rows, args.min_score, args.per_prompt)
    tok = AutoTokenizer.from_pretrained(REPO_ID, cache_dir=CACHE_DIR, revision=REVISION, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    items, skipped = [], 0
    for r in chosen:
        prefix = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True)
        p_ids = tok(prefix, add_special_tokens=False).input_ids
        t_ids = tok(r["text"] + "<|im_end|>", add_special_tokens=False).input_ids
        if len(p_ids) + len(t_ids) > args.max_len:
            skipped += 1
            continue
        items.append({"input_ids": p_ids + t_ids, "labels": [-100] * len(p_ids) + t_ids})
    random.Random(0).shuffle(items)
    print(f"{args.env}: {len(rows)} samples over {len({r['id'] for r in rows})} prompts -> {len(chosen)} selected "
          f"(min_score {args.min_score}), {skipped} over max_len, {len(items)} train items", flush=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(REPO_ID, cache_dir=CACHE_DIR, revision=REVISION, local_files_only=True,
                                                 quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
                                             task_type="CAUSAL_LM", target_modules=TARGETS))
    model.print_trainable_parameters()
    out = Path(args.out)
    cb = LossLog()
    targs = TrainingArguments(output_dir=str(out / "_trainer"), per_device_train_batch_size=args.batch,
                              gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs,
                              learning_rate=args.lr, bf16=True, logging_steps=5, save_strategy="no",
                              optim="paged_adamw_8bit", lr_scheduler_type="cosine", warmup_ratio=0.05,
                              report_to=[], gradient_checkpointing=True, group_by_length=True,
                              remove_unused_columns=False)
    t0 = time.time()
    trainer = Trainer(model=model, args=targs, train_dataset=Rows(items), data_collator=collate(tok.pad_token_id),
                      callbacks=[cb])
    trainer.train()
    wall = time.time() - t0
    model.save_pretrained(str(out))
    print(f"saved adapter to {out} ({wall:.0f}s)", flush=True)

    gguf = out / "lora-f16.gguf"
    env = dict(os.environ, PYTHONPATH=str(Path(LLAMA_SRC) / "gguf-py"))
    subprocess.run([sys.executable, "-X", "utf8", str(Path(LLAMA_SRC) / "convert_lora_to_gguf.py"), str(out),
                    "--base", SNAPSHOT, "--outtype", "f16", "--outfile", str(gguf)], check=True, env=env,
                   stdout=subprocess.DEVNULL)
    print(f"gguf: {gguf} ({gguf.stat().st_size / 1e6:.1f} MB)", flush=True)
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        Path(args.log).write_text(json.dumps({
            "env": args.env, "data": args.data, "n_samples": len(rows), "n_selected": len(chosen),
            "n_train_items": len(items), "skipped_over_max_len": skipped, "min_score": args.min_score,
            "per_prompt": args.per_prompt, "epochs": args.epochs, "lr": args.lr, "rank": args.rank,
            "effective_batch": args.batch * args.grad_accum, "steps": trainer.state.global_step,
            "train_wall_s": round(wall), "loss": cb.hist, "adapter_dir": str(out), "gguf_bytes": gguf.stat().st_size,
        }, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
