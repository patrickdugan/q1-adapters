"""Reference check outside llama.cpp: a community adapter on the REAL Qwen3-8B weights via transformers+PEFT.

  vf venv:    python harness/peft_check.py export --env gsm8k --n 60 --out runs/peft/gsm8k.prompts.jsonl
  torch venv: python harness/peft_check.py generate --prompts ... --adapter <dir or none> --out runs/peft/x.jsonl
  vf venv:    python harness/peft_check.py score --env gsm8k --file runs/peft/x.jsonl

The base is Qwen/Qwen3-8B (bf16 safetensors) loaded in 4-bit NF4 (bitsandbytes) -- a different 4-bit scheme
from llama.cpp's Q4_K_M and a different runtime, with the adapter applied by PEFT exactly as published.
Greedy decoding, Qwen3's own chat template with enable_thinking=False (same prompt text as the servers).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

QWEN3_CACHE = r"C:\Users\patri\Documents\Codex\models\hf"


def cmd_export(a):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from envs import get_env

    env = get_env(a.env)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for ex in env.examples("eval", a.n):
            fh.write(json.dumps({"id": ex.id, "messages": ex.messages, "answer": ex.answer}, ensure_ascii=False) + "\n")


def cmd_generate(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rows = [json.loads(x) for x in Path(a.prompts).read_text(encoding="utf-8").split("\n") if x.strip()]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", cache_dir=QWEN3_CACHE, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", cache_dir=QWEN3_CACHE, local_files_only=True,
                                                 quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.bfloat16)
    if a.adapter and a.adapter != "none":
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    out = []
    for i in range(0, len(rows), a.batch):
        chunk = rows[i:i + a.batch]
        texts = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                 for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        for r, g in zip(chunk, gen):
            text = tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            out.append({**r, "text": text})
        print(f"  {min(i + a.batch, len(rows))}/{len(rows)}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n", encoding="utf-8")


def cmd_score(a):
    import re

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from envs import Example, get_env

    env = get_env(a.env)
    rows = [json.loads(x) for x in Path(a.file).read_text(encoding="utf-8").split("\n") if x.strip()]
    st = re.compile(r"\b(\w{3,})\s+\1\b", re.IGNORECASE)
    scores = [env.score(r["text"], Example(a.env, r["id"], r["messages"], r["answer"]))["score"] for r in rows]
    print(json.dumps({"file": Path(a.file).name, "n": len(rows), "score": round(sum(scores) / len(scores), 3),
                      "stutter": sum(bool(st.search(r["text"])) for r in rows)}))


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export"); e.add_argument("--env", required=True); e.add_argument("--n", type=int, default=60); e.add_argument("--out", required=True)
    g = sub.add_parser("generate"); g.add_argument("--prompts", required=True); g.add_argument("--adapter", default="none")
    g.add_argument("--out", required=True); g.add_argument("--batch", type=int, default=8); g.add_argument("--max-new-tokens", type=int, default=768)
    s = sub.add_parser("score"); s.add_argument("--env", required=True); s.add_argument("--file", required=True)
    a = p.parse_args(argv)
    {"export": cmd_export, "generate": cmd_generate, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
