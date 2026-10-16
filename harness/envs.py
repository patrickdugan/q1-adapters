"""Prime Intellect Environments Hub tasks, loaded from the installed env packages.

Each task is taken from the env's own package: its dataset, its prompt (system
prompt and message layout exactly as the env sends them) and its reward
functions. Only the generation loop is ours, so every configuration -- 1-bit
Bonsai, 4-bit Qwen3, with or without adapters -- is scored by the same code the
Hub environment uses.

Envs (verifiers 0.3.x classic API unless noted):
  reverse-text   primeintellect/reverse-text  LCS ratio of the parsed <reversed_text>
  medmcqa        maziyar/OpenMed_MedMCQA      boxed-letter accuracy (+ env's weighted reward)
  ifeval         arcee-ai/ifeval              RLVR-IFEval constraint check
  gsm8k          primeintellect/gsm8k         math-verify on '#### N' (verifiers v1 package;
                                              its verify.py is run in-process, its prompt read
                                              from the package source)

Runs in the ``vf`` venv (verifiers<0.4 + the four env packages).
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import importlib.util
import inspect
import io
import random
import runpy
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

EVAL_N = 200


@dataclass
class Example:
    env: str
    id: str
    messages: list[dict]
    answer: str
    meta: dict = field(default_factory=dict)


def _call(func, **kw):
    """Call a verifiers reward func with only the kwargs it accepts (sync or async)."""
    sig = inspect.signature(func)
    if not any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        kw = {k: v for k, v in kw.items() if k in sig.parameters}
    out = func(**kw)
    if inspect.isawaitable(out):
        out = asyncio.run(out)
    return float(out)


def _as_messages(prompt) -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in prompt]


class _ClassicEnv:
    """A verifiers classic SingleTurnEnv: dataset 'prompt' column + rubric funcs."""

    name = ""
    primary = ""  # the reward func reported as the task score

    def __init__(self) -> None:
        self.env = self._load()
        self.rubric = self.env.rubric
        # verifiers 0.3.x: the env may wrap its rubric in a RubricGroup with zero-weight monitor metrics
        self.funcs = list(self.rubric._get_reward_funcs())
        self.weights = list(self.rubric._get_reward_weights())

    def _load(self):  # pragma: no cover - per env
        raise NotImplementedError

    def _rows(self, split: str):
        raise NotImplementedError

    def examples(self, split: str, n: int | None = None, seed: int = 0) -> list[Example]:
        rows = self._rows(split)
        idx = list(range(len(rows)))
        if split == "train" and n is not None:
            random.Random(seed).shuffle(idx)
        if n is not None:
            idx = idx[:n]
        out = []
        for i in idx:
            r = rows[i]
            out.append(Example(self.name, f"{split}-{r.get('example_id', i)}", _as_messages(r["prompt"]), str(r["answer"])))
        return out

    def score(self, text: str, ex: Example) -> dict:
        completion = [{"role": "assistant", "content": text}]
        metrics = {}
        reward = 0.0
        for f, w in zip(self.funcs, self.weights):
            try:
                metrics[f.__name__] = _call(f, completion=completion, answer=ex.answer, prompt=ex.messages,
                                            parser=self.env.parser, state={}, info={}, task=self.name)
            except Exception:  # noqa: BLE001 - zero-weight monitor metrics may need rollout state we don't keep
                if w:
                    raise
                continue
            reward += w * metrics[f.__name__]
        return {"score": metrics[self.primary], "reward": reward, "metrics": metrics}


class ReverseText(_ClassicEnv):
    name = "reverse-text"
    primary = "lcs_reward_func"

    def _load(self):
        import reverse_text

        return reverse_text.load_environment()

    def _rows(self, split):
        ds = self.env.dataset  # 1000 rows, train only: hold out the last EVAL_N
        cut = len(ds) - EVAL_N
        return ds.select(range(cut, len(ds))) if split == "eval" else ds.select(range(cut))


class MedMCQA(_ClassicEnv):
    name = "medmcqa"
    primary = "accuracy_reward"

    def _load(self):
        import OpenMed_MedMCQA

        return OpenMed_MedMCQA.load_environment()

    def _rows(self, split):
        return self.env.eval_dataset.select(range(EVAL_N)) if split == "eval" else self.env.dataset


class IFEval(_ClassicEnv):
    name = "ifeval"
    primary = "ifeval_reward_func"

    def _load(self):
        import ifeval

        return ifeval.load_environment()  # eval = the env's own 200-row seeded holdout

    def _rows(self, split):
        return self.env.eval_dataset if split == "eval" else self.env.dataset


class GSM8K:
    """primeintellect/gsm8k is a verifiers-v1 Taskset; reuse its prompt and verify.py verbatim."""

    name = "gsm8k"

    def __init__(self) -> None:
        pkg = Path(importlib.util.find_spec("gsm8k").origin).parent
        tree = ast.parse((pkg / "taskset.py").read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "SYSTEM")
        self.system = ast.literal_eval(node.value)
        self.verify_path = str(pkg / "verify.py")

    def examples(self, split: str, n: int | None = None, seed: int = 0) -> list[Example]:
        from datasets import load_dataset

        rows = load_dataset("openai/gsm8k", "main", split="test" if split == "eval" else "train")
        idx = list(range(len(rows)))
        if split == "train" and n is not None:
            random.Random(seed).shuffle(idx)
        if split == "eval":
            n = EVAL_N if n is None else n
        if n is not None:
            idx = idx[:n]
        out = []
        for i in idx:
            row = rows[i]
            out.append(Example(self.name, f"{split}-{i}",
                               [{"role": "user", "content": f"{self.system}\n\n{row['question']}"}],
                               row["answer"].split("####")[-1].strip()))
        return out

    def score(self, text: str, ex: Example) -> dict:
        # The env's verify.py, line for line, except math-verify's signal/process timeouts are
        # disabled (parsing_timeout=None, timeout_seconds=None): on Windows they spawn a child
        # process that cannot re-run the script, and every answer would score 0.
        import re

        from math_verify import parse, verify

        pred = text or ""
        matches = re.findall(r"####\s*(.+)", pred)
        prediction = matches[-1].strip() if matches else pred
        try:
            s = 1.0 if verify(parse("\\boxed{" + ex.answer + "}", parsing_timeout=None),
                              parse("\\boxed{" + prediction + "}", parsing_timeout=None),
                              timeout_seconds=None) else 0.0
        except Exception:  # noqa: BLE001 - malformed answers can fail anywhere in math-verify
            s = 0.0
        return {"score": s, "reward": s, "metrics": {"correct": s}}


ENVS = {"reverse-text": ReverseText, "medmcqa": MedMCQA, "ifeval": IFEval, "gsm8k": GSM8K}


@lru_cache(maxsize=None)
def get_env(name: str):
    return ENVS[name]()
