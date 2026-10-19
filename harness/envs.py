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
  math500        primeintellect/math500       math-verify on the last \\boxed{} (v1 taskset; prompt and scorer
                                              mirrored from the package, timeouts off as for gsm8k)
  aime           primeintellect/aime2024+2025 math-verify on the boxed answer (v1 tasksets, mirrored)
  pubmedqa       maziyar/OpenMed_PubMedQA     yes/no/maybe accuracy (+ env's weighted reward)
  reasoning-core sileod/reasoning-core-env    the env's symbolic scorer (logic, planning, ...)
  ascii-tree     primeintellect/ascii-tree    the env's weighted tree-similarity reward

Runs in the ``vf`` venv (verifiers<0.4 + the env packages).
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import importlib.util
import inspect
import io
import json
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

    def _rows(self, split: str, n: int | None = None):
        raise NotImplementedError

    def examples(self, split: str, n: int | None = None, seed: int = 0) -> list[Example]:
        rows = self._rows(split, n)
        idx = list(range(len(rows)))
        if split == "train" and n is not None:
            random.Random(seed).shuffle(idx)
        if n is not None:
            idx = idx[:n]
        out = []
        for i in idx:
            r = rows[i]
            info = r.get("info") or {}
            if isinstance(info, str):
                try:
                    info = json.loads(info)
                except ValueError:
                    info = {}
            out.append(Example(self.name, f"{split}-{r.get('example_id', i)}", _as_messages(r["prompt"]), str(r["answer"]),
                               {"info": info}))
        return out

    def score(self, text: str, ex: Example) -> dict:
        completion = [{"role": "assistant", "content": text}]
        metrics = {}
        reward = 0.0
        for f, w in zip(self.funcs, self.weights):
            try:
                metrics[f.__name__] = _call(f, completion=completion, answer=ex.answer, prompt=ex.messages,
                                            parser=self.env.parser, state={}, info=ex.meta.get("info", {}), task=self.name)
            except Exception:  # noqa: BLE001 - zero-weight monitor metrics may need rollout state we don't keep
                if w:
                    raise
                continue
            reward += w * metrics[f.__name__]
        # primary=None: the env's own weighted reward is the task score
        return {"score": reward if self.primary is None else metrics[self.primary], "reward": reward, "metrics": metrics}


class ReverseText(_ClassicEnv):
    name = "reverse-text"
    primary = "lcs_reward_func"

    def _load(self):
        import reverse_text

        return reverse_text.load_environment()

    def _rows(self, split, n=None):
        ds = self.env.dataset  # 1000 rows, train only: hold out the last EVAL_N (fixed; n cannot extend it)
        cut = len(ds) - EVAL_N
        return ds.select(range(cut, len(ds))) if split == "eval" else ds.select(range(cut))


class MedMCQA(_ClassicEnv):
    name = "medmcqa"
    primary = "accuracy_reward"

    def _load(self):
        import OpenMed_MedMCQA

        return OpenMed_MedMCQA.load_environment()

    def _rows(self, split, n=None):
        # eval = the first max(EVAL_N, n) rows of the validation split, so a larger n extends the
        # same prefix (the first 200 stay identical and paired)
        if split == "eval":
            return self.env.eval_dataset.select(range(min(len(self.env.eval_dataset), max(EVAL_N, n or 0))))
        return self.env.dataset


class IFEval(_ClassicEnv):
    name = "ifeval"
    primary = "ifeval_reward_func"

    def _load(self):
        import ifeval

        return ifeval.load_environment()  # eval = the env's own 200-row seeded holdout

    def _rows(self, split, n=None):
        return self.env.eval_dataset if split == "eval" else self.env.dataset  # eval = fixed 200-row holdout


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


def _math_verify_boxed(prediction: str, gold: str) -> float:
    """math-verify equivalence of two boxed expressions, with its signal/process timeouts disabled (Windows)."""
    from math_verify import parse, verify

    if not prediction or not gold:
        return 0.0
    try:
        return 1.0 if verify(parse("\\boxed{" + gold + "}", parsing_timeout=None),
                             parse("\\boxed{" + prediction + "}", parsing_timeout=None), timeout_seconds=None) else 0.0
    except Exception:  # noqa: BLE001 - malformed answers can fail anywhere in math-verify
        return 0.0


def _boxed_v1(text: str, strict: bool = False) -> str:
    """verifiers.v1.utils.score.extract_boxed_answer: contents of the last \\boxed{...}, braces balanced by hand."""
    start = text.rfind("\\boxed{")
    if start == -1:
        return "" if strict else text
    depth, i0 = 1, start + len("\\boxed{")
    for i in range(i0, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[i0:i]
    return "" if strict else text


class Math500:
    """primeintellect/math500 (verifiers-v1 taskset): prompt and scorer mirrored from the package —
    problem + its instruction as one user turn; verify_boxed_math_answer on the last reply."""

    name = "math500"
    EVAL_DEFAULT = 500

    # constants of math500/taskset.py (importing it pulls verifiers.v1, which needs fcntl: POSIX only)
    DATASET, SPLIT, REVISION = "HuggingFaceH4/MATH-500", "test", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
    INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."

    def examples(self, split: str, n: int | None = None, seed: int = 0) -> list[Example]:
        from datasets import load_dataset

        if split != "eval":
            raise ValueError("math500 is eval-only")
        rows = load_dataset(self.DATASET, split=self.SPLIT, revision=self.REVISION)
        k = min(len(rows), n or self.EVAL_DEFAULT)
        return [Example(self.name, f"eval-{i}", [{"role": "user", "content": rows[i]["problem"] + self.INSTRUCTION}],
                        str(rows[i]["answer"])) for i in range(k)]

    def score(self, text: str, ex: Example) -> dict:
        # verifiers.v1.utils.score.verify_boxed_math_answer, line for line, timeouts off
        s = 0.0
        if text and not ("<think>" in text and "</think>" not in text):
            pred = _boxed_v1(text.rsplit("</think>", 1)[-1], strict=True).strip()
            gold = (_boxed_v1(ex.answer, strict=True) or ex.answer).strip()
            s = _math_verify_boxed(pred, gold)
        return {"score": s, "reward": s, "metrics": {"correct": s}}


class Aime:
    """primeintellect/aime2024 + aime2025 (verifiers-v1 tasksets), 60 problems: their instruction + problem as the
    user turn; MaybeThinkParser(extract_boxed_answer) then math-verify, as in the packages."""

    name = "aime"
    INSTR = "Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}.\n\n"

    def examples(self, split: str, n: int | None = None, seed: int = 0) -> list[Example]:
        from datasets import concatenate_datasets, load_dataset

        if split != "eval":
            raise ValueError("aime is eval-only")
        out = []
        a24 = load_dataset("HuggingFaceH4/aime_2024", split="train", revision="2fe88a2f1091d5048c0f36abc874fb997b3dd99a")
        for i, r in enumerate(a24):
            out.append(Example(self.name, f"eval-2024-{i}", [{"role": "user", "content": self.INSTR + r["problem"]}],
                               str(int(r["answer"]))))
        a25 = concatenate_datasets([load_dataset("opencompass/AIME2025", c, split="test",
                                                 revision="a6ad95f611d72cf628a80b58bd0432ef6638f958")
                                    for c in ("AIME2025-I", "AIME2025-II")])
        for i, r in enumerate(a25):
            ans = "".join(c for c in r["answer"] if c.isdigit() or c == ".")
            out.append(Example(self.name, f"eval-2025-{i}", [{"role": "user", "content": self.INSTR + r["question"]}], ans))
        return out[:n] if n else out

    def score(self, text: str, ex: Example) -> dict:
        import verifiers as vf
        from verifiers.utils.data_utils import extract_boxed_answer

        resp = vf.MaybeThinkParser(extract_boxed_answer).parse_answer([{"role": "assistant", "content": text or ""}])
        s = 0.0 if (not resp or len(resp) > 50_000) else _math_verify_boxed(resp, ex.answer)
        return {"score": s, "reward": s, "metrics": {"correct": s}}


class PubMedQA(_ClassicEnv):
    name = "pubmedqa"
    primary = "accuracy_reward"

    def _load(self):
        import OpenMed_PubMedQA

        return OpenMed_PubMedQA.load_environment()

    def _rows(self, split, n=None):
        if split == "eval":
            return self.env.eval_dataset.select(range(min(len(self.env.eval_dataset), max(500, n or 0))))
        return self.env.dataset


class ReasoningCore(_ClassicEnv):
    name = "reasoning-core"
    primary = "score_answer_vf"

    def _load(self):
        import reasoning_core_env

        return reasoning_core_env.load_environment(num_train_examples=800, num_eval_examples=500, seed=0)

    def _rows(self, split, n=None):
        return self.env.eval_dataset if split == "eval" else self.env.dataset


class AsciiTree(_ClassicEnv):
    name = "ascii-tree"
    primary = None  # the env's weighted reward (0.3 similarity + 0.7 continuous)

    def _load(self):
        import ascii_tree

        return ascii_tree.load_environment()

    def _rows(self, split, n=None):
        ds = self.env.dataset  # single split: hold out the last EVAL_N rows
        cut = len(ds) - EVAL_N
        return ds.select(range(cut, len(ds))) if split == "eval" else ds.select(range(cut))


ENVS = {"reverse-text": ReverseText, "medmcqa": MedMCQA, "ifeval": IFEval, "gsm8k": GSM8K,
        "math500": Math500, "aime": Aime, "pubmedqa": PubMedQA, "reasoning-core": ReasoningCore, "ascii-tree": AsciiTree}


@lru_cache(maxsize=None)
def get_env(name: str):
    return ENVS[name]()
