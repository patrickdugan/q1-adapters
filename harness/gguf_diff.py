"""Compare two GGUF files: metadata keys/values that differ, tensor-name sets, and layer-0 tensor shapes/types.

  python harness/gguf_diff.py a.gguf b.gguf
"""

import sys

from gguf import GGUFReader


def info(path):
    r = GGUFReader(path)
    meta = {}
    for k, f in r.fields.items():
        if k.startswith("tokenizer.ggml.") and k not in ("tokenizer.ggml.model", "tokenizer.ggml.pre"):
            continue
        try:
            v = f.contents()
        except Exception:  # noqa: BLE001
            v = "?"
        meta[k] = v if not isinstance(v, list) or len(v) < 20 else f"<list {len(v)}>"
    tensors = {t.name: (tuple(int(x) for x in t.shape), t.tensor_type.name) for t in r.tensors}
    return meta, tensors


a, b = sys.argv[1], sys.argv[2]
ma, ta = info(a)
mb, tb = info(b)
print("== metadata differing")
for k in sorted(set(ma) | set(mb)):
    if ma.get(k) != mb.get(k):
        print(f"  {k}: A={str(ma.get(k))[:90]} | B={str(mb.get(k))[:90]}")
print(f"== tensors: A {len(ta)}  B {len(tb)}")
print("  only in A:", sorted(set(ta) - set(tb))[:12])
print("  only in B:", sorted(set(tb) - set(ta))[:12])
print("== blk.0 + global tensors (A | B)")
for n in sorted(set(ta) | set(tb)):
    if n.startswith("blk.0.") or not n.startswith("blk."):
        print(f"  {n:28s} A={ta.get(n)}  B={tb.get(n)}")
