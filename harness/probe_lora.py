"""Probe one adapter on a running llama-server with a PER-REQUEST LoRA override (does not touch the
server's global adapter scales, so a concurrent run is unaffected apart from sharing the GPU).

  python harness/probe_lora.py --env gsm8k --lora-id 1 --n 30 --temperature 0 --port 8096
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from envs import get_env  # noqa: E402

STUTTER = re.compile(r"\b(\w{3,})\s+\1\b", re.IGNORECASE)


async def main_async(a):
    env = get_env(a.env)
    exs = env.examples("eval", a.n)
    client = AsyncOpenAI(base_url=f"http://127.0.0.1:{a.port}/v1", api_key="local", timeout=900)
    sem = asyncio.Semaphore(a.concurrency)
    lora = [{"id": i, "scale": (1.0 if i == a.lora_id else 0.0)} for i in range(a.n_loras)] if a.lora_id >= 0 else \
        [{"id": i, "scale": 0.0} for i in range(a.n_loras)]

    async def one(ex):
        async with sem:
            r = await client.chat.completions.create(
                model="local", messages=ex.messages, temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens,
                extra_body={"top_k": a.top_k, "seed": 1, "lora": lora})
            t = r.choices[0].message.content or ""
            return env.score(t, ex)["score"], r.choices[0].finish_reason, bool(STUTTER.search(t))

    res = await asyncio.gather(*(one(ex) for ex in exs))
    out = {"env": a.env, "lora_id": a.lora_id, "temperature": a.temperature, "n": len(res),
           "score": round(sum(s for s, _, _ in res) / len(res), 3),
           "truncated": sum(f == "length" for _, f, _ in res), "stutter": sum(st for _, _, st in res)}
    print(json.dumps(out))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="gsm8k")
    p.add_argument("--lora-id", type=int, default=-1)
    p.add_argument("--n-loras", type=int, default=8)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--port", type=int, default=8096)
    p.add_argument("--concurrency", type=int, default=2)
    asyncio.run(main_async(p.parse_args(argv)))


if __name__ == "__main__":
    main()
