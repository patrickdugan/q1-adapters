"""Compare F32 norm tensors between GGUFs (AWQ-style rescaling shows up as changed norm weights) and print names.

  python harness/gguf_norms.py a.gguf b.gguf [c.gguf ...]
"""

import sys

import numpy as np
from gguf import GGUFReader

NAMES = ["blk.0.attn_norm.weight", "blk.0.ffn_norm.weight", "blk.17.attn_norm.weight", "blk.17.ffn_norm.weight",
         "blk.0.attn_q_norm.weight", "output_norm.weight"]
files = sys.argv[1:]
data = {}
for f in files:
    r = GGUFReader(f)
    name = r.fields["general.name"].contents() if "general.name" in r.fields else None
    fin = r.fields["general.finetune"].contents() if "general.finetune" in r.fields else None
    print(f"{f.split(chr(92))[-1]}: general.name={name!r} finetune={fin!r}")
    data[f] = {t.name: np.array(t.data, dtype=np.float32).ravel() for t in r.tensors if t.name in NAMES}
ref = files[0]
for n in NAMES:
    row = [f"{n:26s}"]
    for f in files:
        v = data[f].get(n)
        if v is None:
            row.append("—")
            continue
        rel = np.abs(v - data[ref][n]).max() / (np.abs(data[ref][n]).max() + 1e-12)
        row.append(f"mean {v.mean():8.4f} max {v.max():8.3f} rel-diff-vs-first {rel:.3f}")
    print("  " + " | ".join(row))
