"""Aggregate results/<env>/*.jsonl into the paper's tables and figures.

  python harness/make_tables.py            (Spiral .venv or any env with numpy + matplotlib)

Outputs results/summary.json, results/tables.md, results/figures/score_vs_memory.png.
Paired deltas: every configuration answers the same eval examples with the same
per-example seed, so differences are computed per example and bootstrapped.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path(__file__).resolve().parents[1]
RES = R / "results"
ENVS = ["reverse-text", "gsm8k", "medmcqa", "ifeval"]
METRIC = {"reverse-text": "LCS ratio", "gsm8k": "accuracy", "medmcqa": "accuracy", "ifeval": "constraint pass rate"}
GB = 1e9
Q1_BYTES = 1_158_654_496
Q4_BYTES = 5_027_783_488
Q4S_BYTES = 1_107_409_472  # Qwen3-1.7B Q4_K_M: the memory-matched 4-bit baseline
Q4OWN_BYTES = 5_027_783_616  # our Q4_K_M from the exact Qwen/Qwen3-8B weights
LORA_BYTES = 87_328_928


def load(env: str, stem: str) -> dict[str, float] | None:
    f = RES / env / f"{stem}.jsonl"
    if not f.exists():
        return None
    rows = [json.loads(x) for x in f.read_text(encoding="utf-8").split("\n") if x.strip()]  # not splitlines (U+2028)
    return {r["id"]: r["score"] for r in rows if r.get("k", 0) == 0}


def mean(d):
    return sum(d.values()) / len(d) if d else float("nan")


def boot_ci(xs, n=2000, seed=0):
    rng = random.Random(seed)
    ms = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def paired(a: dict, b: dict):
    ids = sorted(set(a) & set(b))
    if not ids:
        return None
    diffs = [b[i] - a[i] for i in ids]
    lo, hi = boot_ci(diffs)
    return {"n": len(ids), "delta": sum(diffs) / len(diffs), "ci95": [lo, hi]}


def best_online(env: str, prefix: str):
    best = None
    for f in sorted((RES / env).glob(f"{prefix}online-*.jsonl")):
        d = load(env, f.stem)
        if d and len(d) < 200:  # an in-flight (incomplete) run would shrink the row's common item set
            continue
        if d and (best is None or mean(d) > mean(best[1])):
            best = (f.stem, d)
    return best


def load_avg(env: str, stem: str) -> tuple[dict[str, float], dict[str, float]] | None:
    """Per-id mean over all k samples (avg@k) and per-id best (pass@k)."""
    f = RES / env / f"{stem}.jsonl"
    if not f.exists():
        return None
    by: dict[str, list[float]] = {}
    for x in f.read_text(encoding="utf-8").split("\n"):
        if x.strip():
            r = json.loads(x)
            by.setdefault(r["id"], []).append(r["score"])
    return {i: sum(v) / len(v) for i, v in by.items()}, {i: max(v) for i, v in by.items()}


# Phase N: larger samples on two original envs and five more Hub envs. Reasoning load rises down the table.
PHASE_N = [("gsm8k", "accuracy, full test", "q1+ours"), ("medmcqa", "accuracy", "q1+ours"),
           ("pubmedqa", "accuracy", "q1+ours"), ("ascii-tree", "weighted tree similarity", "q1+ours"),
           ("math500", "accuracy", "q1+ours-gsm8k"), ("reasoning-core", "task score", "q1+ours"),
           ("aime", "avg@2; pass@2 in brackets", None)]
PN_COLS = {"q4-1.7b-base": "Q4 1.7B base (1.11 GB)", "q1-base": "Q1 8B base (1.16 GB)",
           "ours": "Q1 + our adapter", "q4own-base": "Q4 8B base (5.03 GB)"}


def phase_n(lines: list[str]) -> dict:
    out = {}
    lines += ["", "Phase N: bigger samples and more Prime Intellect Hub environments, ordered roughly by reasoning load "
              "(non-thinking mode; every row averaged over the ids all of its configurations answered):", "",
              "| env (metric) | n | " + " | ".join(PN_COLS.values()) + " | Q1 − Q4 1.7B | Q1 − Q4 8B |",
              "|---|---|" + "---|" * (len(PN_COLS) + 2)]
    for env, metric, ours in PHASE_N:
        stems = {"q4-1.7b-base": "q4-1.7b-base", "q1-base": "q1-base", "ours": ours, "q4own-base": "q4own-base"}
        ds, best = {}, {}
        for col, stem in stems.items():
            if stem and (got := load_avg(env, stem)):
                ds[col], best[col] = got
        # a run still in flight would shrink the common set: require the base columns to be complete
        if not ds:
            continue
        need = max(len(d) for d in ds.values())
        ds = {c: d for c, d in ds.items() if len(d) >= 0.98 * need}
        best = {c: best[c] for c in ds}
        common = set.intersection(*(set(d) for d in ds.values()))
        ds = {c: {i: d[i] for i in common} for c, d in ds.items()}
        best = {c: {i: d[i] for i in common} for c, d in best.items()}
        row = {"n": len(common), "metric": metric}

        def cell(c):
            if c not in ds:
                return "—"
            s = f"{mean(ds[c]):.3f}"
            return s + (f" [{mean(best[c]):.3f}]" if env == "aime" else "")

        def dcell(a, b):
            if a in ds and b in ds:
                d = paired(ds[a], ds[b])
                row[f"delta {b} vs {a}"] = d
                return f"{d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
            return "—"
        for c in ds:
            row[c] = mean(ds[c])
            if env == "aime":
                row[c + " pass@2"] = mean(best[c])
        d17, d8 = dcell("q4-1.7b-base", "q1-base"), dcell("q4own-base", "q1-base")
        if "ours" in ds and "q1-base" in ds:
            dcell("q1-base", "ours")
        lines.append(f"| {env} ({metric}) | {len(common)} | " + " | ".join(cell(c) for c in PN_COLS) + f" | {d17} | {d8} |")
        out[env] = row
    lines.append("")
    lines.append("Our adapters in phase N: the GSM8K adapter on gsm8k / math500 / aime, the MedMCQA adapter on medmcqa, "
                 "phase N2 adapters on pubmedqa (rejection sampling), ascii-tree and reasoning-core (gold SFT); "
                 "paired deltas vs Q1 base:")
    for env, row in out.items():
        d = row.get("delta ours vs q1-base")
        if d:
            lines.append(f"- {env}: {d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] (n={d['n']})")
    if out:
        fig, ax = plt.subplots(figsize=(9, 4.0))
        cols = [("q4-1.7b-base", "#6b4c9a"), ("q1-base", "#9aa5b1"), ("ours", "#1f6feb"), ("q4own-base", "#c9a227")]
        envs = list(out)
        w = 0.2
        for j, (c, colr) in enumerate(cols):
            xs = [i + (j - 1.5) * w for i, e in enumerate(envs) if c in out[e]]
            ys = [out[e][c] for e in envs if c in out[e]]
            ax.bar(xs, ys, w, color=colr, label=PN_COLS[c])
        ax.set_xticks(range(len(envs)), [e if e != "aime" else "aime (avg@2)" for e in envs], rotation=0, fontsize=8)
        ax.set_ylabel("score")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1))
        ax.set_title("More Hub environments, ordered by reasoning load (non-thinking mode)", fontsize=10)
        fig.tight_layout()
        (RES / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(RES / "figures" / "phase_n_envs.png", dpi=150)
    return out


def main() -> int:
    summary, lines = {}, []
    lines.append("| env (metric) | n | Q4 1.7B base (1.11 GB) | Q1 8B base (1.16 GB) | Q1 + our adapter (1.25 GB) | "
                 "Q1 + online Qwen3 adapter | Q4 8B base (5.03 GB) | Q4 8B + online adapter (5.11 GB) | Q4 8B + our Bonsai adapter |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    pts = []
    official = []
    for env in ENVS:
        # Q4 8B columns come from our own Q4_K_M converted from the exact Qwen/Qwen3-8B weights ("q4own") when
        # available: Qwen's official Q4_K_M GGUF is built from an AWQ-rescaled checkpoint (different weight basis),
        # which misapplies LoRAs trained on Qwen/Qwen3-8B. The official-file results are kept as a separate block.
        p = "q4own" if (RES / env / "q4own-base.jsonl").exists() else "q4"
        q1b, q1o, q4b = load(env, "q1-base"), load(env, "q1+ours"), load(env, f"{p}-base")
        q4s = load(env, "q4-1.7b-base")
        q1on, q4on = best_online(env, "q1+"), best_online(env, f"{p}+")
        q4ours = load(env, f"{p}+bonsai-adapter")
        if p == "q4own":
            ob, oon, oours = load(env, "q4-base"), best_online(env, "q4+"), load(env, "q4+bonsai-adapter")
            official.append((env, ob and mean(ob), oon and (oon[0], mean(oon[1])), oours and mean(oours)))
        # every cell of a row is averaged over the SAME examples: the ids all its configurations answered
        present = [d for d in (q1b, q1o, q4b, q4s, q4ours, q1on and q1on[1], q4on and q4on[1]) if d]
        common = set.intersection(*(set(d) for d in present)) if present else set()

        def cut(d):
            return d and {k: v for k, v in d.items() if k in common}
        q1b, q1o, q4b, q4s, q4ours = cut(q1b), cut(q1o), cut(q4b), cut(q4s), cut(q4ours)
        q1on = q1on and (q1on[0], cut(q1on[1]))
        q4on = q4on and (q4on[0], cut(q4on[1]))
        row = {"n_common": len(common), "q4_1.7b_base": q4s and mean(q4s), "q1_base": q1b and mean(q1b), "q1_ours": q1o and mean(q1o),
               "q4_base": q4b and mean(q4b),
               "q1_online": q1on and {"adapter": q1on[0], "score": mean(q1on[1])},
               "q4_online": q4on and {"adapter": q4on[0], "score": mean(q4on[1])},
               "q4_bonsai_adapter": q4ours and mean(q4ours)}
        if q1b and q1o:
            row["delta_q1_ours_vs_q1_base"] = paired(q1b, q1o)
        if q1b and q4s:
            row["delta_q1_base_vs_q4_1.7b_base"] = paired(q4s, q1b)
        if q1b and q1on:
            row["delta_q1_online_vs_q1_base"] = paired(q1b, q1on[1])
        if q4b and q4on:
            row["delta_q4_online_vs_q4_base"] = paired(q4b, q4on[1])
        if q1o and q4b:
            row["delta_q1_ours_vs_q4_base"] = paired(q4b, q1o)
        if q1o and q4on:
            row["delta_q1_ours_vs_q4_online"] = paired(q4on[1], q1o)
        summary[env] = row

        def f(x):
            return "—" if x is None else f"{x:.3f}"

        def fo(o):
            return "—" if not o else f"{mean(o[1]):.3f} ({o[0].split('online-')[-1]})"
        lines.append(f"| {env} ({METRIC[env]}) | {row['n_common']} | {f(row['q4_1.7b_base'])} | {f(row['q1_base'])} | {f(row['q1_ours'])} | "
                     f"{fo(q1on)} | {f(row['q4_base'])} | {fo(q4on)} | {f(row['q4_bonsai_adapter'])} |")
        q4mem = Q4OWN_BYTES if p == "q4own" else Q4_BYTES
        for lab, val, mem in [("Q4 1.7B base", row["q4_1.7b_base"], Q4S_BYTES),
                              ("Q1 base", row["q1_base"], Q1_BYTES), ("Q1 + adapter", row["q1_ours"], Q1_BYTES + LORA_BYTES),
                              ("Q4 base", row["q4_base"], q4mem),
                              ("Q4 + online adapter", q4on and mean(q4on[1]), q4mem + LORA_BYTES)]:
            if val is not None:
                pts.append((env, lab, mem / GB, val))

    if official:
        lines += ["", "Q4 8B columns above: our own Q4_K_M from the exact Qwen/Qwen3-8B weights. Qwen's official "
                  "Qwen3-8B-Q4_K_M.gguf (general.name 'Qwen3 8B Awq Compatible Instruct'; AWQ-rescaled norms) for comparison:", "",
                  "| env | official Q4_K_M base | official + best online adapter | official + our Bonsai adapter |",
                  "|---|---|---|---|"]
        for env, ob, oon, oours in official:
            base_s = "—" if ob is None else f"{ob:.3f}"
            online_s = "—" if not oon else f"{oon[1]:.3f} ({oon[0].split('online-')[-1]})"
            ours_s = "—" if oours is None else f"{oours:.3f}"
            lines.append(f"| {env} | {base_s} | {online_s} | {ours_s} |")
    lines.append("")
    lines.append("Paired deltas (same examples, same seeds; bootstrap 95% CI):")
    for env, row in summary.items():
        for k in ("delta_q1_base_vs_q4_1.7b_base", "delta_q1_online_vs_q1_base", "delta_q1_ours_vs_q1_base",
                  "delta_q4_online_vs_q4_base", "delta_q1_ours_vs_q4_base", "delta_q1_ours_vs_q4_online"):
            d = row.get(k)
            if d:
                lines.append(f"- {env} {k.replace('delta_', '').replace('_', ' ')}: {d['delta']:+.3f} "
                             f"[{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] (n={d['n']})")
    # extended table: gsm8k + medmcqa at n=500 for the configurations run that far (common ids per env)
    ext = {"q4-1.7b-base": "Q4 1.7B base", "q1-base": "Q1 base", "q1+ours": "Q1 + our adapter",
           "q1+online-gsm8k-grpo": "Q1 + GRPO-GSM8K", "q1+online-gsm8k-metamath": "Q1 + MetaMathQA",
           "q1+online-medmcqa-vireshk": "Q1 + MedMCQA LoRA", "q4-base": "Q4 8B base"}
    lines += ["", "Extended evaluation, 500 examples (configurations run to 500; each row averaged over the ids all of "
              "its present configurations share):", "",
              "| env | n | " + " | ".join(ext.values()) + " |", "|---|---|" + "---|" * len(ext)]
    ext_summary = {}
    for env in ("gsm8k", "medmcqa"):
        ds = {k: load(env, k) for k in ext}
        ds = {k: d for k, d in ds.items() if d and len(d) >= 500}
        if not ds:
            continue
        common = set.intersection(*(set(d) for d in ds.values()))
        ds = {k: {i: d[i] for i in common} for k, d in ds.items()}
        lines.append(f"| {env} | {len(common)} | " + " | ".join(f"{mean(ds[k]):.3f}" if k in ds else "—" for k in ext) + " |")
        ext_summary[env] = {k: mean(d) for k, d in ds.items()}
        for a, b in (("q1-base", "q1+ours"), ("q4-1.7b-base", "q1-base"), ("q4-1.7b-base", "q1+ours"), ("q4-base", "q1+ours")):
            if a in ds and b in ds:
                d = paired(ds[a], ds[b])
                ext_summary[env][f"delta {b} vs {a}"] = d
                lines.append(f"- {env} (n={d['n']}) {ext[b]} vs {ext[a]}: {d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]")
    summary["_extended_n500"] = ext_summary
    summary["_phase_n"] = phase_n(lines)

    # speed: llama-bench (no adapter) and served throughput with/without an active adapter
    sp = RES / "speed"
    bench_rows = []
    for name in ("q1", "q4-1.7b", "q4"):
        for dev in ("gpu", "cpu"):
            f = sp / f"{name}-{dev}.json"
            if not f.exists():
                continue
            try:
                for b in json.loads(f.read_text(encoding="utf-8-sig") or "[]"):
                    kind = "pp" if b.get("n_prompt") else "tg"
                    bench_rows.append((name, dev, kind, b.get("n_prompt") or b.get("n_gen"), b.get("avg_ts"),
                                       b.get("model_size")))
            except json.JSONDecodeError:
                pass
    if bench_rows:
        lines += ["", "llama-bench (PrismML fork build 10743; GPU = RTX 5080 Laptop, -ngl 99; CPU = 8 threads):", "",
                  "| model | device | test | tokens/s | model size (GB) |", "|---|---|---|---|---|"]
        for name, dev, kind, n, ts, size in bench_rows:
            lines.append(f"| {name} | {dev} | {kind}{n} | {ts:.1f} | {size / 1e9:.2f} |")
    memf = RES / "memory.json"
    if memf.exists():
        mem = json.loads(memf.read_text(encoding="utf-8-sig"))
        lines += ["", "Deployment footprint (llama-server, 1 slot, 4k context; VRAM = nvidia-smi delta, RAM = working set):", "",
                  "| server | weights file | adapters | VRAM | process RAM |", "|---|---|---|---|---|"]
        for m in mem:
            lines.append(f"| {m['config'].replace('mem-', '')} | {m['model_bytes'] / 1e9:.2f} GB | {m['lora_bytes'] / 1e6:.0f} MB | "
                         f"{m['vram_delta_mib'] / 1024:.2f} GiB | {m['ram_working_set_mib'] / 1024:.2f} GiB |")
    thr = []
    for env in ENVS:
        for f in sorted((RES / env).glob("*.summary.json")):
            s = json.loads(f.read_text(encoding="utf-8"))
            if s.get("agg_tokens_per_s_this_run"):
                thr.append((env, s["label"], s["agg_tokens_per_s_this_run"]))
    if thr:
        lines += ["", "Served throughput during the eval runs (6 concurrent requests, aggregate decode tokens/s):", "",
                  "| env | configuration | tokens/s |", "|---|---|---|"]
        lines += [f"| {e} | {lab} | {t:.0f} |" for e, lab, t in thr]
    (RES / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RES / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if pts:
        (RES / "figures").mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, len(ENVS), figsize=(4 * len(ENVS), 3.6), sharex=True)
        style = {"Q4 1.7B base": ("#6b4c9a", "^"), "Q1 base": ("#9aa5b1", "o"), "Q1 + adapter": ("#1f6feb", "o"),
                 "Q4 base": ("#c9a227", "s"), "Q4 + online adapter": ("#d1495b", "s")}
        for ax, env in zip(axes, ENVS):
            for e, lab, mem, val in pts:
                if e == env:
                    c, m = style[lab]
                    ax.scatter(mem, val, color=c, marker=m, s=60, label=lab, zorder=3)
            ax.set_title(f"{env}")
            ax.set_xlabel("weights on disk / in VRAM (GB)")
            ax.set_ylabel(METRIC[env])
            ax.grid(alpha=0.3)
            ax.set_xlim(0, 6)
        seen = {}
        for ax in axes:  # union over panels: reverse-text has no online adapter, so its legend alone is short
            for h, lab in zip(*ax.get_legend_handles_labels()):
                seen.setdefault(lab, h)
        fig.legend(list(seen.values()), list(seen.keys()), loc="lower center", ncol=5, frameon=False)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(RES / "figures" / "score_vs_memory.png", dpi=150)
    print((RES / "tables.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
