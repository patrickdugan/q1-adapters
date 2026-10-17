"""Export routing data: which env (= which adapter) a prompt belongs to.

  python harness/export_router_data.py --out runs/router/data.jsonl

Two views of each prompt:
  full     -- the messages exactly as the env sends them (system prompt + user turn)
  content  -- the user's content only, with the env's own instruction scaffolding removed
              (gsm8k's instruction prefix; reverse-text's and medmcqa's system prompts).
              This is the realistic routing problem: a bare sentence, a word problem, a
              multiple-choice item or a constrained request, with no label in the prompt.

Train rows come from each env's TRAIN split (n per env, seeded); eval rows are exactly the
eval examples the adapters were scored on (ids match results/<env>/*.jsonl).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from envs import ENVS, get_env  # noqa: E402


def content_only(env_name: str, env, messages: list[dict]) -> str:
    user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
    if env_name == "gsm8k":
        prefix = env.system + "\n\n"
        return user[len(prefix):] if user.startswith(prefix) else user
    return user  # system prompts carry the instructions for the other envs; the user turn is the content


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n-train", type=int, default=800)
    p.add_argument("--n-eval", type=int, default=200)
    args = p.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as fh:
        for name in ENVS:
            env = get_env(name)
            for split, k in (("train", args.n_train), ("eval", args.n_eval)):
                exs = env.examples(split, k, seed=7)
                for ex in exs:
                    full = "\n\n".join(f"[{m['role']}] {m['content']}" for m in ex.messages)
                    fh.write(json.dumps({"env": name, "split": split, "id": ex.id, "full": full,
                                         "content": content_only(name, env, ex.messages)}, ensure_ascii=False) + "\n")
                    n += 1
                print(f"{name} {split}: {len(exs)}", flush=True)
    print(f"wrote {n} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
