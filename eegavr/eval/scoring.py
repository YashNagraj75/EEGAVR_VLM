import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from .parse import parse_answer

PANEL_CHANCE, PUZZLE_CHANCE = 0.5, 0.25
GROUPINGS = (
    ("model", "condition", "layout", "split"),
    ("model", "condition", "layout", "split", "type"),
    ("model", "condition", "layout", "split", "rule_id"),
)


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def binom_sf(k, n, p0):
    # One-sided exact P(X >= k) for X ~ Binomial(n, p0), summed in log space so large n cannot overflow.
    if n == 0:
        return float("nan")
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    lp, lq = math.log(p0), math.log1p(-p0)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
            for i in range(k, n + 1)]
    top = max(logs)
    return min(1.0, math.exp(top) * sum(math.exp(v - top) for v in logs))


def score_record(rec):
    # Re-parses raw output, so parser fixes apply retroactively to old logs.
    if rec.get("status") == "error":
        parsed = {"T1": None, "T2": None, "rule": None, "status": "error"}
    else:
        parsed = parse_answer(rec.get("raw_output", ""))
    truth = rec["truth"]
    c1, c2 = parsed["T1"] == truth["T1"], parsed["T2"] == truth["T2"]
    return {**rec, "parsed": parsed, "parse_status": parsed["status"], "t1_correct": c1, "t2_correct": c2,
            "both_correct": c1 and c2}


def load_records(paths):
    # Last record per key wins (a retried error replaces the failed one).
    recs = {}
    for path in paths:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(r, dict) and "key" in r and "truth" in r:
                    recs[r["key"]] = r
    return [score_record(r) for r in recs.values()]


def summarize(records, keys):
    groups = defaultdict(list)
    for r in records:
        groups[tuple(r.get(k) for k in keys)].append(r)
    rows = []
    for gk, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        n = len(rs)
        kp = sum(r["t1_correct"] + r["t2_correct"] for r in rs)
        kb = sum(r["both_correct"] for r in rs)
        status = defaultdict(int)
        for r in rs:
            status[r["parse_status"]] += 1
        answers = [r["parsed"][t] for r in rs for t in ("T1", "T2") if r["parsed"][t]]
        lo_p, hi_p = wilson(kp, 2 * n)
        lo_b, hi_b = wilson(kb, n)
        row = dict(zip(keys, gk))
        row.update({
            "n_puzzles": n, "n_panels": 2 * n,
            "panel_acc": kp / (2 * n), "panel_ci_lo": lo_p, "panel_ci_hi": hi_p, "panel_chance": PANEL_CHANCE,
            "panel_p_above_chance": binom_sf(kp, 2 * n, PANEL_CHANCE),
            "both_acc": kb / n, "both_ci_lo": lo_b, "both_ci_hi": hi_b, "both_chance": PUZZLE_CHANCE,
            "both_p_above_chance": binom_sf(kb, n, PUZZLE_CHANCE),
            **{f"n_{s}": status[s] for s in ("json", "json_repaired", "regex", "partial", "unparsable", "error")},
            "strict_json_rate": status["json"] / n,
            "frac_left": (sum(a == "left" for a in answers) / len(answers)) if answers else float("nan"),
            "frac_same_side": sum(r["parsed"]["T1"] is not None and r["parsed"]["T1"] == r["parsed"]["T2"]
                                  for r in rs) / n,
            "mean_prompt_tokens": _mean([r.get("n_prompt_tokens") for r in rs]),
            "mean_output_tokens": _mean([r.get("n_output_tokens") for r in rs]),
            "mean_seconds": _mean([r.get("seconds") for r in rs]),
        })
        rows.append(row)
    return rows


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else float("nan")


def _boot_ci(diffs, n_boot=2000, seed=0):
    if not diffs:
        return (float("nan"), float("nan"))
    rng, n = random.Random(seed), len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1])


def b_vs_partner_a(records):
    # Pairs each Type B problem with its A partner (same base scenes) within model/condition/layout.
    idx = {(r["model"], r["condition"], r["layout"], r["problem_id"]): r for r in records}
    groups = defaultdict(list)
    for r in records:
        if r.get("type") != "B" or not r.get("partner_problem_id"):
            continue
        a = idx.get((r["model"], r["condition"], r["layout"], r["partner_problem_id"]))
        if a is not None:
            groups[(r["model"], r["condition"], r["layout"], r["split"], r["rule_id"], a["rule_id"])].append((r, a))
    rows = []
    for (model, cond, lay, split, rb, ra), pairs in sorted(groups.items()):
        n = len(pairs)
        pb = [(b["t1_correct"] + b["t2_correct"]) / 2 for b, _ in pairs]
        pa = [(a["t1_correct"] + a["t2_correct"]) / 2 for _, a in pairs]
        bb = [float(b["both_correct"]) for b, _ in pairs]
        ba = [float(a["both_correct"]) for _, a in pairs]
        dp = [x - y for x, y in zip(pb, pa)]
        db = [x - y for x, y in zip(bb, ba)]
        lo_p, hi_p = _boot_ci(dp)
        lo_b, hi_b = _boot_ci(db)
        rows.append({
            "model": model, "condition": cond, "layout": lay, "split": split, "rule_B": rb, "rule_A": ra,
            "n_pairs": n, "panel_acc_B": sum(pb) / n, "panel_acc_A": sum(pa) / n, "panel_diff_B_minus_A": sum(dp) / n,
            "panel_diff_ci_lo": lo_p, "panel_diff_ci_hi": hi_p,
            "both_acc_B": sum(bb) / n, "both_acc_A": sum(ba) / n, "both_diff_B_minus_A": sum(db) / n,
            "both_diff_ci_lo": lo_b, "both_diff_ci_hi": hi_b,
            "mcnemar_B_only": sum(x > y for x, y in zip(bb, ba)), "mcnemar_A_only": sum(y > x for x, y in zip(bb, ba)),
        })
    return rows


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = []
    for r in rows:
        cols.extend(k for k in r if k not in cols)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    return path


def score_dir(out_dir):
    out_dir = Path(out_dir)
    paths = sorted(p for p in out_dir.rglob("*.jsonl") if p.name != "run_manifest.jsonl")
    records = load_records(paths)
    summary = []
    for keys in GROUPINGS:
        for row in summarize(records, keys):
            summary.append({"level": "+".join(keys[4:]) or "stage", **row})
    write_csv(summary, out_dir / "summary.csv")
    pairs = b_vs_partner_a(records)
    write_csv(pairs, out_dir / "b_vs_partner_a.csv")
    return records, summary, pairs
