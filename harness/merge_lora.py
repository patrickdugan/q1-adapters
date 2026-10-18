"""Merge a PEFT LoRA into HF safetensors weights, streaming tensor by tensor (low RAM).

  python harness/merge_lora.py --base <hf snapshot dir> --adapter <peft dir> --out <dir>

W' = W + (alpha / r) * B @ A for every adapted Linear (fp32 math, cast back to the base dtype).
Output is re-sharded into ~1 GB files with a fresh model.safetensors.index.json; config and tokenizer
files are copied. Used to test whether a LoRA that fails at runtime on a Q4_K_M GGUF also fails when
merged into the weights BEFORE quantization (i.e. runtime-LoRA path vs. the quantized weights).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SHARD_BYTES = 1_000_000_000


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    base, ad, out = Path(a.base), Path(a.adapter), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ad / "adapter_config.json").read_text(encoding="utf-8"))
    scale = cfg["lora_alpha"] / cfg["r"]
    if cfg.get("use_rslora"):
        scale = cfg["lora_alpha"] / cfg["r"] ** 0.5
    lora = {}
    with safe_open(str(ad / "adapter_model.safetensors"), "pt") as f:
        for k in f.keys():
            lora[k] = f.get_tensor(k)
    # map base weight name -> (A, B); PEFT keys: base_model.model.<name minus .weight>.lora_{A,B}.weight
    pairs = {}
    for k, t in lora.items():
        if ".lora_A." in k or ".lora_B." in k:
            stem = k.replace("base_model.model.", "", 1).split(".lora_")[0] + ".weight"
            pairs.setdefault(stem, {})["A" if ".lora_A." in k else "B"] = t
    print(f"adapter: r={cfg['r']} alpha={cfg['lora_alpha']} scale={scale} modules={len(pairs)}", flush=True)

    index = json.loads((base / "model.safetensors.index.json").read_text(encoding="utf-8"))
    names = sorted(index["weight_map"], key=lambda n: (index["weight_map"][n], n))
    shard, shard_bytes, n_shard, weight_map, merged = {}, 0, 0, {}, 0
    handles = {}

    def flush():
        nonlocal shard, shard_bytes, n_shard
        if not shard:
            return
        n_shard += 1
        fname = f"model-merged-{n_shard:05d}.safetensors"
        save_file(shard, str(out / fname), metadata={"format": "pt"})
        for k in shard:
            weight_map[k] = fname
        shard, shard_bytes = {}, 0

    for name in names:
        fn = index["weight_map"][name]
        if fn not in handles:
            handles[fn] = safe_open(str(base / fn), "pt")
        t = handles[fn].get_tensor(name)
        if name in pairs:
            A, B = pairs[name]["A"].float(), pairs[name]["B"].float()
            t = (t.float() + scale * (B @ A)).to(t.dtype)
            merged += 1
        shard[name] = t.contiguous()
        shard_bytes += t.numel() * t.element_size()
        if shard_bytes >= SHARD_BYTES:
            flush()
    flush()
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": index.get("metadata", {}), "weight_map": weight_map},
                                                                 indent=2), encoding="utf-8")
    for f in base.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
            shutil.copy2(f, out / f.name)
    print(f"merged {merged}/{len(pairs)} modules into {n_shard} shards at {out}", flush=True)
    return 0 if merged == len(pairs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
