"""MedMCQA: env score vs format-agnostic accuracy (the chosen letter, wherever it appears).

The env's parser only credits \\boxed{X}. Community adapters trained on MedMCQA-style data answer with a
bare letter first ("A\\n\\nexplanation"), which the env scores as wrong. This separates knowledge from format.
"""

import json
import re
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results" / "medmcqa"
BOXED = re.compile(r"\\boxed\{([A-Da-d])\}")
LEAD = re.compile(r"^\s*(?:</?think>\s*)*\(?([A-D])\)?(?:[.:)\s]|$)")


def letter(t):
    m = BOXED.findall(t)
    if m:
        return m[-1].upper()
    m = LEAD.match(t)
    return m.group(1) if m else None


stems = sys.argv[1:] or ["q1-base", "q1+ours", "q1+online-medmcqa-vireshk", "q4-base", "q4+online-medmcqa-vireshk",
                         "q4own-base", "q4own+online-medmcqa-vireshk", "q4-1.7b-base"]
for stem in stems:
    f = R / f"{stem}.jsonl"
    if not f.exists():
        continue
    rows = [json.loads(x) for x in f.read_text(encoding="utf-8").split("\n") if x.strip()][:200]
    env = sum(r["score"] for r in rows) / len(rows)
    fa = sum(letter(r["text"]) == r["answer"] for r in rows) / len(rows)
    un = sum(letter(r["text"]) is None for r in rows)
    print(f"{stem:32s} env-score {env:.3f}   format-agnostic {fa:.3f}   no letter found {un}/{len(rows)}")
