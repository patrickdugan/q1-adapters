"""Export an env's TRAIN split with gold targets as SFT rows (for envs whose answer is the full target).

  python harness/export_gold.py --env reverse-text --out runs/sft/reverse-text-gold.jsonl

reverse-text: the target is the env's own ground truth (the reversed string) in the
<reversed_text> tags its XMLParser reads -- the same shape as Prime Intellect's SFT
warm-up for this env. Held-out eval rows are never exported (envs.py keeps them apart).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from envs import get_env  # noqa: E402

TARGETS = {"reverse-text": lambda ans: f"<reversed_text>\n{ans}\n</reversed_text>",
           # ascii-tree: the gold tree in the <ascii_formatted> tags its XMLParser reads (a transformation skill)
           "ascii-tree": lambda ans: f"<ascii_formatted>\n{ans}\n</ascii_formatted>",
           # reasoning-core: the gold answer alone, no derivation -- tests whether a format-level adapter buys reasoning
           "reasoning-core": lambda ans: f"<answer>{ans}</answer>"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=sorted(TARGETS))
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=None, help="first N train rows only")
    args = p.parse_args(argv)
    env = get_env(args.env)
    rows = env.examples("train")[: args.n]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for ex in rows:
            text = TARGETS[args.env](ex.answer)
            sc = env.score(text, ex)
            fh.write(json.dumps({"id": ex.id, "k": 0, "messages": ex.messages, "answer": ex.answer, "text": text,
                                 "label": "gold", **sc}, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} gold rows to {out} (eval rows held out: {len(env.examples('eval'))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
