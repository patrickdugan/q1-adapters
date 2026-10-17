"""Route prompts to task adapters on the 1-bit base: TRM, LDT, and TRM/LDT hybrids.

  python harness/routers.py            (torch venv, CPU is enough)

Follows the TRM/LDT split of Dugan, "Trade Offs between TRM/LDT Hybrids" (2026):

* TRM -- private recursive proposer. A tiny recursive classifier in the style of
  Jolicoeur-Martineau's Tiny Recursive Model: hashed word/char features -> x; a latent z and
  an answer state y refined by two weight-shared 2-layer MLPs (n inner latent steps per outer
  step, T outer steps), deep supervision on every outer step.
* LDT -- explicit lattice state. Slots = {env}; each known word w carries the candidate set
  E_w = {env : w occurs in >= m training prompts of env}; the prompt's state is the meet
  C(x) = /\\_w E_w (unknown words are top). C = {} is bottom.
* Hybrids (control policies from the paper):
    soft         argmax_{e in C} TRM if TRM's top route is in C, else TRM's top route (lexical
                 lattice evidence is not environment-sound, so it is soft telemetry)
    hard-filter  argmax_{e in C} TRM whenever C is non-bottom
    arbitration  TRM's top route if its margin >= gamma, else argmax_{e in C} TRM (gamma from CV)
    membrane     TRM's top route unless it conflicts with a non-bottom lattice (top route not in C)
                 and its margin < gamma_a -- then ABSTAIN: serve the request on the base model with
                 no adapter. Abstention is the safe action here because a wrong adapter is
                 catastrophic (the reverse-text adapter takes gsm8k from 0.85 to 0.06).

Scored two ways: routing accuracy, and END-TO-END task score -- for every eval prompt we look
up the env score of the generation produced under the chosen adapter (results/<env>/q1+ours*.jsonl,
q1-base.jsonl), so routing is judged by what it does to the answers, not just by the label.
"""

from __future__ import annotations

import json
import math
import random
import re
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

R = Path(__file__).resolve().parents[1]
ENVS = ["reverse-text", "gsm8k", "medmcqa", "ifeval"]
WORD = re.compile(r"[a-z0-9]+")
BUCKETS, EMB, D = 2048, 24, 48


# ---------------------------------------------------------------- data
def load_rows():
    rows = [json.loads(x) for x in (R / "runs/router/data.jsonl").read_text(encoding="utf-8").split("\n") if x.strip()]
    return [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "eval"]


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def feats(text: str) -> list[int]:
    w = words(text)
    grams = w + [a + "_" + b for a, b in zip(w, w[1:])]
    t = text.lower()
    grams += ["#" + t[i:i + 3] for i in range(0, max(0, len(t) - 2), 2)]
    return [zlib.crc32(g.encode()) % BUCKETS for g in grams] or [0]


# ---------------------------------------------------------------- TRM
class TRM(nn.Module):
    def __init__(self, n_cls: int, n_inner: int = 3, n_outer: int = 3):
        super().__init__()
        self.emb = nn.EmbeddingBag(BUCKETS, EMB, mode="mean")
        self.inp = nn.Linear(EMB, D)
        self.f = nn.Sequential(nn.Linear(3 * D, 64), nn.GELU(), nn.Linear(64, D))  # latent update z <- z + f(x,y,z)
        self.g = nn.Sequential(nn.Linear(2 * D, 64), nn.GELU(), nn.Linear(64, D))  # answer update y <- y + g(y,z)
        self.head = nn.Linear(D, n_cls)
        self.n_inner, self.n_outer = n_inner, n_outer

    def forward(self, ids, offsets):
        x = self.inp(self.emb(ids, offsets))
        y = torch.zeros_like(x)
        z = torch.zeros_like(x)
        outs = []
        for _ in range(self.n_outer):
            for _ in range(self.n_inner):
                z = z + self.f(torch.cat([x, y, z], -1))
            y = y + self.g(torch.cat([y, z], -1))
            outs.append(self.head(y))
        return outs  # deep supervision: one prediction per outer step


_FEATS: dict[str, list[int]] = {}


def batchify(texts):
    ids, offs = [], []
    for t in texts:
        offs.append(len(ids))
        f = _FEATS.get(t)
        if f is None:
            f = _FEATS[t] = feats(t)
        ids.extend(f)
    return torch.tensor(ids), torch.tensor(offs)


def train_trm(texts, labels, seed=0, epochs=25):
    torch.manual_seed(seed)
    m = TRM(len(ENVS))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-4)
    y = torch.tensor(labels)
    idx = list(range(len(texts)))
    rng = random.Random(seed)
    for _ in range(epochs):
        rng.shuffle(idx)
        for i in range(0, len(idx), 64):
            b = idx[i:i + 64]
            outs = m(*batchify([texts[j] for j in b]))
            loss = sum(F.cross_entropy(o, y[b]) for o in outs) / len(outs)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m


@torch.no_grad()
def trm_probs(m, texts):
    return F.softmax(m(*batchify(texts))[-1], -1)


# ---------------------------------------------------------------- LDT
class LDT:
    def __init__(self, texts, labels, min_df: int = 2):
        df = defaultdict(Counter)
        for t, l in zip(texts, labels):
            for w in set(words(t)):
                df[w][l] += 1
        self.E = {w: frozenset(e for e, c in cnt.items() if c >= min_df) for w, cnt in df.items()}
        self.E = {w: s for w, s in self.E.items() if s}
        self.top = frozenset(range(len(ENVS)))

    def state(self, text):
        c = self.top
        for w in set(words(text)):
            s = self.E.get(w)
            if s is not None:  # unknown word = top, no information
                c = c & s
        return c  # frozenset() is bottom

    def votes(self, text):
        v = Counter()
        for w in set(words(text)):
            s = self.E.get(w)
            if s is not None and len(s) < len(ENVS):
                for e in s:
                    v[e] += 1.0 / len(s)
        return v


# ---------------------------------------------------------------- policies
ABSTAIN = -1  # serve on the base model with no adapter


def route_all(p, C, votes, gamma, gamma_a):
    top = int(p.argmax())
    srt = sorted(p.tolist(), reverse=True)
    margin = srt[0] - srt[1]
    in_c = sorted(C, key=lambda e: -float(p[e]))
    ldt = (max(C, key=lambda e: votes.get(e, 0.0)) if C else ABSTAIN)
    return {
        "trm": top,
        "ldt": ldt,
        "soft": (in_c[0] if top in C else top) if C else top,
        "hard-filter": in_c[0] if C else top,
        "arbitration": top if (margin >= gamma or not C) else in_c[0],
        "membrane": top if (not C or top in C or margin >= gamma_a) else ABSTAIN,
        "trm-abstain": top if srt[0] >= 0.5 + gamma_a / 2 else ABSTAIN,
    }


def cv_gamma(texts, labels, policy_key, seeds=3, grid=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)):
    """Pick gamma on the TRAIN split by 4-fold cross-validated routing accuracy (abstain counts wrong)."""
    folds = 4
    order = list(range(len(texts)))
    random.Random(1).shuffle(order)
    score = Counter()
    for k in range(folds):
        va = set(order[k::folds])
        tr = [i for i in order if i not in va]
        m = train_trm([texts[i] for i in tr], [labels[i] for i in tr], seed=k, epochs=15)
        ldt = LDT([texts[i] for i in tr], [labels[i] for i in tr])
        va_list = [i for i in order if i in va]
        P = trm_probs(m, [texts[i] for i in va_list])
        for p, i in zip(P, va_list):
            C, v = ldt.state(texts[i]), ldt.votes(texts[i])
            for g in grid:
                score[g] += route_all(p, C, v, g, g)[policy_key] == labels[i]
    return max(grid, key=lambda g: (score[g], -g))


THRESHOLD_POLICIES = {
    # name -> (p, C, gamma) -> choice; gamma is chosen for END-TO-END score by 2-fold cross-fitting
    "membrane (e2e-tuned)": lambda p, C, g: (lambda top, m: top if (not C or top in C or m >= g) else ABSTAIN)(
        int(p.argmax()), float(p.sort(descending=True).values[0] - p.sort(descending=True).values[1])),
    "trm-abstain (e2e-tuned)": lambda p, C, g: int(p.argmax()) if float(p.max()) >= g else ABSTAIN,
}
GRID = (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01)


def e2e_macro(choices, rows, S):
    per_env = defaultdict(list)
    for c, row in zip(choices, rows):
        env = row["env"]
        per_env[env].append(S[env]["base" if c == ABSTAIN else ENVS[c]][row["id"]])
    return {e: sum(v) / len(v) for e, v in per_env.items()}, (sum(sum(v) / len(v) for v in per_env.values()) / len(per_env))


def eu_crossfit(P, rows, S):
    """Bayes-decision routing: choose the action (an adapter, or the base model) that maximizes
    expected env score sum_e P(e|x) M[e][a], where M is the cross-task score matrix estimated on the
    OTHER half of the prompts (cross-fitted, so no prompt's own outcome informs its route)."""
    acts = list(range(len(ENVS))) + [ABSTAIN]
    halves = [[i for i, r in enumerate(rows) if zlib.crc32(r["id"].encode() + r["env"].encode()) % 2 == h] for h in (0, 1)]
    chosen = {}
    for h in (0, 1):
        tune, test = halves[h], halves[1 - h]
        M = {}
        for e in ENVS:
            ids = [rows[i]["id"] for i in tune if rows[i]["env"] == e]
            for a in acts:
                cfg = "base" if a == ABSTAIN else ENVS[a]
                M[(e, a)] = sum(S[e][cfg][x] for x in ids) / len(ids)
        for i in test:
            p = P[i]
            chosen[i] = max(acts, key=lambda a: sum(float(p[k]) * M[(ENVS[k], a)] for k in range(len(ENVS))))
    order = sorted(chosen)
    return [chosen[i] for i in order], [rows[i] for i in order]


def confusion(choices, rows):
    m = defaultdict(Counter)
    for c, r in zip(choices, rows):
        m[r["env"]]["abstain" if c == ABSTAIN else ENVS[c]] += 1
    return {e: dict(m[e]) for e in ENVS}


def crossfit(P, states, rows, S, policy):
    """Pick gamma on one half of the e2e prompts, score on the other half, and vice versa."""
    halves = [[i for i, r in enumerate(rows) if zlib.crc32(r["id"].encode() + r["env"].encode()) % 2 == h] for h in (0, 1)]
    chosen, picks = {}, []
    for h in (0, 1):
        tune, test = halves[h], halves[1 - h]
        best = max(GRID, key=lambda g: e2e_macro([policy(P[i], states[i], g) for i in tune], [rows[i] for i in tune], S)[1])
        picks.append(best)
        for i in test:
            chosen[i] = policy(P[i], states[i], best)
    order = sorted(chosen)
    return [chosen[i] for i in order], [rows[i] for i in order], picks


# ---------------------------------------------------------------- end-to-end scores
def score_table():
    """S[env][config][id] where config in {'base', <adapter env>}; from the 1-bit runs."""
    S = defaultdict(dict)
    for env in ENVS:
        def rd(stem):
            f = R / "results" / env / f"{stem}.jsonl"
            if not f.exists():
                return None
            return {json.loads(x)["id"]: json.loads(x)["score"] for x in f.read_text(encoding="utf-8").split("\n") if x.strip()}
        S[env]["base"] = rd("q1-base")
        for a in ENVS:
            S[env][a] = rd("q1+ours") if a == env else rd(f"q1+ours-{a}")
    return S


def main() -> int:
    train, ev = load_rows()
    S = score_table()
    report = {"params": None, "views": {}}
    lines = []
    for view in ("full", "content"):
        tx = [r[view] for r in train]
        ty = [ENVS.index(r["env"]) for r in train]
        ex = [r[view] for r in ev]
        eyy = [ENVS.index(r["env"]) for r in ev]
        g_arb = cv_gamma(tx, ty, "arbitration")
        g_mem = cv_gamma(tx, ty, "membrane")
        accs, e2e, abst, gam, conf = defaultdict(list), defaultdict(list), defaultdict(list), {}, {}
        for seed in range(5):  # router seeds; LDT is deterministic
            m = train_trm(tx, ty, seed=seed)
            report["params"] = sum(p.numel() for p in m.parameters())
            ldt = LDT(tx, ty)
            P = trm_probs(m, ex)
            choice = defaultdict(list)
            for p, t in zip(P, ex):
                r = route_all(p, ldt.state(t), ldt.votes(t), g_arb, g_mem)
                for k, v in r.items():
                    choice[k].append(v)
            for k, ch in choice.items():
                accs[k].append(sum(c == y for c, y in zip(ch, eyy)) / len(eyy))
                abst[k].append(sum(c == ABSTAIN for c in ch) / len(ch))
                # end-to-end: score of the answer produced under the chosen adapter (abstain -> base)
                per_env = defaultdict(list)
                for c, row in zip(ch, ev):
                    env = row["env"]
                    cfg = "base" if c == ABSTAIN else ENVS[c]
                    tbl = S[env].get(cfg)
                    if tbl is None or row["id"] not in tbl or any(S[env].get(a) is None or row["id"] not in S[env][a]
                                                                  for a in ENVS + ["base"]):
                        continue  # only prompts where every adapter's answer exists
                    per_env[env].append(tbl[row["id"]])
                e2e[k].append({e: sum(v) / len(v) for e, v in per_env.items() if v})
            # threshold policies tuned for end-to-end score (cross-fitted on the e2e prompts)
            ok = [i for i, row in enumerate(ev)
                  if all(S[row["env"]].get(a) is not None and row["id"] in S[row["env"]][a] for a in ENVS + ["base"])]
            if ok:
                Pk = [P[i] for i in ok]
                Ck = [ldt.state(ex[i]) for i in ok]
                rk = [ev[i] for i in ok]
                for name, pol in THRESHOLD_POLICIES.items():
                    ch, rr, picks = crossfit(Pk, Ck, rk, S, pol)
                    accs[name].append(sum(c == ENVS.index(r["env"]) for c, r in zip(ch, rr)) / len(rr))
                    abst[name].append(sum(c == ABSTAIN for c in ch) / len(ch))
                    e2e[name].append(e2e_macro(ch, rr, S)[0])
                    gam.setdefault(name, []).append(picks)
                ch, rr = eu_crossfit(Pk, rk, S)
                name = "trm + expected-utility (Bayes)"
                accs[name].append(sum(c == ENVS.index(r["env"]) for c, r in zip(ch, rr)) / len(rr))
                abst[name].append(sum(c == ABSTAIN for c in ch) / len(ch))
                e2e[name].append(e2e_macro(ch, rr, S)[0])
                if seed == 0:
                    conf.setdefault("trm", confusion([choice["trm"][i] for i in ok], rk))
                    conf.setdefault(name, confusion(ch, rr))
        # reference policies on the same prompts
        refs = {}
        for name, pick in [("oracle", lambda env: env), ("no adapter", lambda env: "base")] + \
                          [(f"always {a}", (lambda a: (lambda env: a))(a)) for a in ENVS]:
            per_env = defaultdict(list)
            for row in ev:
                env = row["env"]
                if any(S[env].get(a) is None or row["id"] not in S[env][a] for a in ENVS + ["base"]):
                    continue
                per_env[env].append(S[env][pick(env)][row["id"]])
            refs[name] = {e: sum(v) / len(v) for e, v in per_env.items() if v}
        n_e2e = {e: len(v) for e, v in per_env.items()}
        res = {"gamma_arbitration": g_arb, "gamma_membrane": g_mem, "n_e2e_per_env": n_e2e, "policies": {}, "refs": refs,
               "e2e_tuned_gamma_picks": gam, "confusion_seed0": conf}
        for k in accs:
            macro = [sum(d.values()) / len(d) for d in e2e[k] if d]
            res["policies"][k] = {"route_acc_mean": round(sum(accs[k]) / len(accs[k]), 4),
                                  "route_acc_min": round(min(accs[k]), 4),
                                  "abstain_rate": round(sum(abst[k]) / len(abst[k]), 4),
                                  "e2e_macro_mean": round(sum(macro) / len(macro), 4) if macro else None,
                                  "e2e_per_env": {e: round(sum(d[e] for d in e2e[k]) / len(e2e[k]), 4) for e in (e2e[k][0] if e2e[k] else {})}}
        report["views"][view] = res
        lines.append(f"\n### view: {view}  (TRM params {report['params']:,}; gamma_arb={g_arb}, gamma_membrane={g_mem}; "
                     f"end-to-end on {sum(n_e2e.values())} prompts {n_e2e})\n")
        lines.append("| router | routing acc (5 seeds) | abstain | end-to-end macro score | " + " | ".join(ENVS) + " |")
        lines.append("|---|---|---|---|" + "---|" * len(ENVS))
        for name, d in refs.items():
            mac = sum(d.values()) / len(d) if d else float("nan")
            lines.append(f"| {name} | — | — | {mac:.3f} | " + " | ".join(f"{d.get(e, float('nan')):.3f}" for e in ENVS) + " |")
        for k, v in res["policies"].items():
            pe = v["e2e_per_env"]
            mac = v["e2e_macro_mean"] if v["e2e_macro_mean"] is not None else float("nan")
            lines.append(f"| {k} | {v['route_acc_mean']:.3f} (min {v['route_acc_min']:.3f}) | {v['abstain_rate']:.3f} | {mac:.3f} | "
                         + " | ".join(f"{pe.get(e, float('nan')):.3f}" for e in ENVS) + " |")
        for cname, cm in conf.items():
            lines.append(f"\nRoutes chosen by {cname} (seed 0; rows = true env):\n")
            cols = ENVS + ["abstain"]
            lines.append("| true env | " + " | ".join(cols) + " |")
            lines.append("|---|" + "---|" * len(cols))
            for e in ENVS:
                lines.append(f"| {e} | " + " | ".join(str(cm[e].get(c, 0)) for c in cols) + " |")
    (R / "results" / "routing.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (R / "results" / "routing.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
