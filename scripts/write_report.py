import argparse
import glob
import json
import os
import shutil
import time
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eegavr.generator.settings import audit_tier


def f3(x):
    return "–" if x is None else f"{x:.3f}"


def ci(c):
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


def setting_rows(audit_dir):
    rows = []
    for f in sorted(glob.glob(f"{audit_dir}/*.json")):
        r = json.load(open(f))
        a, fu, dg, b = r["5a"]["pass_set"], r["5a"]["full_set"], r["5a"]["diagnostic_with_abs_role_centroids"], r["5b"]
        u = r.get("c_univariate")
        d = r["5d"]
        k = lambda s: d[str(s)] if str(s) in d else d[s]
        uni = "" if not u else (f"pixel worst {u['pixel']['worst_cue']} {f3(u['pixel']['worst_acc'])} "
                                f"{ci(u['pixel']['worst_ci90'])}; cue worst {u['symbolic']['worst_cue']} "
                                f"{f3(u['symbolic']['worst_acc'])} {ci(u['symbolic']['worst_ci90'])}")
        rows.append((r["setting_id"], r["type"], f"{r['knob_min']:.3f}–{r['knob_max']:.3f}", f"{r['kept_rate']:.3f}",
                     f"{a['best']} {f3(a['best_acc'])} {ci(a['best_ci90'])}", f"{f3(fu['best_acc'])}",
                     f"{f3(dg['best_acc'])}", f"{b['best']} {f3(b['best_acc'])} {ci(b['best_ci90'])}", uni,
                     f"{k(224)['all_ok']:.3f}/{k(448)['all_ok']:.3f}", audit_tier(r).upper()))
    return rows


def main():
    # audit/report.md plus per-setting JSON and the full run log copied into the exported set
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--audit", required=True)
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--notes", required=True)
    a = ap.parse_args()
    out = f"{a.root}/audit"
    os.makedirs(out, exist_ok=True)
    for f in glob.glob(f"{a.audit}/*.json"):
        shutil.copy(f, out)
    with open(f"{out}/audit_runs_log.jsonl", "w") as g:
        for lf in a.logs:
            for line in open(lf):
                d = json.loads(line)
                d["log_file"] = os.path.relpath(lf, os.path.dirname(a.audit))
                g.write(json.dumps(d) + "\n")
    man = json.load(open(f"{a.root}/manifest.json"))
    rc = json.load(open(f"{out}/rendering_check.json")) if os.path.exists(f"{out}/rendering_check.json") else None
    L = [f"# Audit report — EEGAVR pilot stimuli ({time.strftime('%Y-%m-%d')})", "",
         f"Generator {man['generator_version']}, contract {man['contract_version']}, git commit {man['git_commit']}.", ""]
    L += open(a.notes).read().splitlines() + [""]
    L += ["## Per-setting audit (1,000 twin pairs = 2,000 panels each)", "",
          "Columns: 5a pass probe = role-free global pixel features (criterion for B); full = adds role-relative "
          "and region features (diagnostic); +abs = adds absolute role centroids (diagnostic, encodes the relation); "
          "5b = symbolic leakage probe; C univariate = criterion for C; 5d = share of panels passing all rendering "
          "checks at 224/448 px (bicubic, setting level).", "",
          "| setting | type | knob range | kept | 5a pass probe (best, acc, 90% CI) | 5a full | 5a +abs | 5b | C univariate | 5d 224/448 | result |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in setting_rows(a.audit):
        L.append("| " + " | ".join(r) + " |")
    L += ["", "## Problem assembly and 5c (competing rules)", "",
          "| rule × split | problems | 5c attempts | 5c resampled | resample rate | partner attempts | partner resampled | kept draws |",
          "|---|---|---|---|---|---|---|---|"]
    for k, v in sorted(man["assembly_stats"].items()):
        L.append(f"| {k} | {v['problems']} | {v['5c_attempts']} | {v['5c_rejected']} | {v['5c_resample_rate']:.3f} | "
                 f"{v['partner_5c_attempts']} | {v['partner_5c_rejected']} | {v['kept_rate']:.3f} |")
    if rc:
        L += ["", "## Per-panel rendering check on the exported set (5d, PIL BOX)", "",
              f"{rc['n_panels']} panels; problems with any failing panel: {rc['n_problems_failing']}.", "",
              "| size | no clip | components | gaps ≥ 2 px | stroke ≥ 1 px | min gap (px) | failing panels |",
              "|---|---|---|---|---|---|---|"]
        for s, v in rc["summary"].items():
            L.append(f"| {s} | {v['no_clip']:.4f} | {v['components_ok']:.4f} | {v['gap_ok']:.4f} | "
                     f"{v['stroke_ok']:.4f} | {v['min_gap_px']:.2f} | {v['panels_failing']} |")
    L += ["", "## Audit run history (every run, every generator version)", "",
          "| time | log | generator | setting | pairs | kept | 5a pass-set (best, acc, CI) | 5a full | 5b | C univariate worst | pass |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for line in open(f"{out}/audit_runs_log.jsonl"):
        d = json.loads(line)
        u = d.get("c_univariate_worst")
        us = "" if not u else "; ".join(f"{k}: {v[0]} {v[1]:.3f}" for k, v in u.items())
        L.append(f"| {d['time']} | {d['log_file']} | {d['generator_version']} | {d['setting_id']} | {d['n_pairs']} | "
                 f"{d['kept_rate']:.3f} | {d['5a_pass_set'][0]} {d['5a_pass_set'][1]:.3f} {ci(d['5a_pass_set'][2])} | "
                 f"{d['5a_full_set'][1]:.3f} | {d['5b'][1]:.3f} | {us} | {d['pass']} |")
    open(f"{out}/report.md", "w").write("\n".join(L) + "\n")
    print("wrote", f"{out}/report.md", len(L), "lines")


if __name__ == "__main__":
    main()
