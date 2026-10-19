"""Run a Prime Intellect env against an OpenAI-compatible endpoint (llama-server) and score it.

  eval:  python harness/generate.py --env gsm8k --split eval --label q1-base --out results/gsm8k/q1-base
  train: python harness/generate.py --env gsm8k --split train --n 600 --k 4 --label q1-base-rft --out runs/...

Writes <out>.jsonl (one row per sample: messages, text, score, reward, metrics, tokens,
latency) and <out>.summary.json (mean score with a 95% CI, reward, token and
throughput stats). Sampling follows Qwen3's non-thinking recommendation
(T=0.7, top-p 0.8, top-k 20) with a per-example seed, so every configuration is
asked the same questions with the same seeds (paired comparisons).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from envs import get_env  # noqa: E402

MAX_TOKENS = {"reverse-text": 384, "gsm8k": 768, "medmcqa": 1024, "ifeval": 1024,
              "math500": 2048, "aime": 6144, "pubmedqa": 1024, "reasoning-core": 1024, "ascii-tree": 768}


def _seed(ex_id: str, k: int) -> int:
    return int(hashlib.sha256(f"{ex_id}/{k}".encode()).hexdigest()[:8], 16)


def _lines(path: Path) -> list[str]:
    """JSONL records: split on '\\n' only. str.splitlines() also splits on U+2028/U+2029/U+0085,
    which json.dumps(ensure_ascii=False) leaves raw inside strings, and would cut a record in two."""
    # errors="replace": a crash can leave a torn tail with invalid bytes; that line then fails json and is dropped
    return [x for x in path.read_text(encoding="utf-8", errors="replace").split("\n") if x.strip()]


def _ci95(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return 1.96 * statistics.stdev(xs) / math.sqrt(len(xs))


async def _one(client, sem, args, ex, k, max_tokens):
    async with sem:
        t0 = time.time()
        extra = {"top_k": args.top_k, "seed": _seed(ex.id, k),
                 "chat_template_kwargs": {"enable_thinking": False}}
        for attempt in range(4):
            try:
                resp = await client.chat.completions.create(
                    model=args.model, messages=ex.messages, temperature=args.temperature,
                    top_p=args.top_p, max_tokens=max_tokens, extra_body=extra)
                break
            except Exception as exc:  # noqa: BLE001 - transient server errors
                if attempt == 3:
                    raise
                print(f"  retry {ex.id}/{k}: {type(exc).__name__}", file=sys.stderr, flush=True)
                await asyncio.sleep(5 * (attempt + 1))
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = resp.usage
        return {"id": ex.id, "k": k, "messages": ex.messages, "answer": ex.answer, "text": text,
                "finish_reason": choice.finish_reason,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "latency_s": round(time.time() - t0, 3)}


async def _run(args) -> int:
    env = get_env(args.env)
    exs = env.examples(args.split, args.n, seed=args.seed)
    max_tokens = args.max_tokens or MAX_TOKENS[args.env]
    client = AsyncOpenAI(base_url=args.base_url, api_key="local", timeout=900)
    sem = asyncio.Semaphore(args.concurrency)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = out.parent / (out.name + ".jsonl")  # not with_suffix: a stem like 'q4-1.7b-base' has a dot in it
    done = set()
    if jsonl.exists() and args.resume:
        good, bad = [], 0
        for line in _lines(jsonl):
            try:
                r = json.loads(line)
                done.add((r["id"], r["k"]))
                good.append(line)
            except ValueError:  # a crash mid-write leaves a torn or NUL-filled tail; drop it and rerun those items
                bad += 1
        if bad:
            jsonl.write_text("".join(x + "\n" for x in good), encoding="utf-8")
            print(f"resume: dropped {bad} unreadable line(s) from {jsonl.name}", flush=True)
    jobs = [(ex, k) for ex in exs for k in range(args.k) if (ex.id, k) not in done]
    print(f"{args.env}/{args.split}: {len(exs)} examples x k={args.k}; {len(jobs)} to run ({len(done)} resumed)", flush=True)
    t0 = time.time()
    with jsonl.open("a", encoding="utf-8") as fh:
        tasks = [asyncio.create_task(_one(client, sem, args, ex, k, max_tokens)) for ex, k in jobs]
        exmap = {ex.id: ex for ex in exs}
        for i, fut in enumerate(asyncio.as_completed(tasks), 1):
            row = await fut
            row.update(env.score(row["text"], exmap[row["id"]]))
            row["label"] = args.label
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    wall = time.time() - t0

    rows = [json.loads(line) for line in _lines(jsonl)]
    scores = [r["score"] for r in rows]
    toks = [r["completion_tokens"] or 0 for r in rows]
    summary = {
        "env": args.env, "split": args.split, "label": args.label, "n_examples": len(exs), "k": args.k,
        "n_samples": len(rows), "score_mean": round(statistics.fmean(scores), 4), "score_ci95": round(_ci95(scores), 4),
        "reward_mean": round(statistics.fmean(r["reward"] for r in rows), 4),
        "metrics_mean": {m: round(statistics.fmean(r["metrics"][m] for r in rows), 4) for m in rows[0]["metrics"]},
        "completion_tokens_mean": round(statistics.fmean(toks), 1),
        "truncated_frac": round(sum(r["finish_reason"] == "length" for r in rows) / len(rows), 4),
        "wall_s_this_run": round(wall, 1), "agg_tokens_per_s_this_run": round(sum(toks) / wall, 1) if jobs and wall else None,
        "sampling": {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
                     "max_tokens": max_tokens, "enable_thinking": False, "concurrency": args.concurrency},
        "base_url": args.base_url,
    }
    (out.parent / (out.name + ".summary.json")).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("env", "label", "n_samples", "score_mean", "score_ci95", "reward_mean",
                                               "completion_tokens_mean", "truncated_frac", "agg_tokens_per_s_this_run")}), flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env", required=True)
    p.add_argument("--split", default="eval", choices=["eval", "train"])
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True, help="output path stem")
    p.add_argument("--base-url", default="http://127.0.0.1:8096/v1")
    p.add_argument("--model", default="local")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--resume", action="store_true")
    return asyncio.run(_run(p.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
