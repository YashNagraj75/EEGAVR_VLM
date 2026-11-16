import argparse
import json
import os
import subprocess
import sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eegavr.generator.assemble import build_job
from eegavr.generator.export import scene_json, write_panel, ensure_dirs
from eegavr.generator.settings import all_settings, RULE_TYPE, RULE_PROGRAM, KNOB_NAME, PARTNERS, TUNING
from eegavr.generator.version import GENERATOR_VERSION, CONTRACT_VERSION

SPLITS = {"validation": 50, "test_id": 100, "ood_count": 100, "ood_count_control": 100, "ood_render": 100}


def _write(args):
    root, pid, sj, sc = args
    write_panel(root, pid, sj, sc)
    return pid


def main():
    # build, 5c-check and export the pilot set per docs/DATA_CONTRACT.md
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--rules", default="A1,A6,B1,B3,B4,C1")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    root = a.out
    ensure_dirs(root)
    S = all_settings()
    eq = {sid: np.load(f"{a.audit}/{sid}_pool.npz")["eq"] for sid in S if os.path.exists(f"{a.audit}/{sid}_pool.npz")}
    jobs = [(r, sp, max(1, int(n * a.scale)), eq, {}) for r in a.rules.split(",") for sp, n in SPLITS.items()
            if not (r == "C1" and sp == "ood_count")]
    with Pool(a.workers) as pool:
        res = pool.map(build_job, jobs, chunksize=1)
        tasks = []
        for j in res:
            for pid, sc, seed, regen in j["panels"]:
                tasks.append((root, pid, scene_json(sc, pid, seed, regen), sc))
        for _ in pool.imap_unordered(_write, tasks, chunksize=32):
            pass
    splits = {sp: [] for sp in SPLITS}
    stats = {}
    for j in res:
        for p in j["problems"]:
            with open(f"{root}/problems/{p['problem_id']}.json", "w") as f:
                json.dump(p, f)
            splits[p["split"]].append(p["problem_id"])
        s, ps = j["stats"], j["partner_stats"]
        stats[f"{j['rule']}|{j['split']}"] = {
            "problems": sum(1 for p in j["problems"] if p["problem_id"].startswith(j["rule"] + "-")),
            "5c_attempts": s["attempts"], "5c_rejected": s["rejected"],
            "5c_resample_rate": s["rejected"] / max(s["attempts"], 1), "kept_rate": s["pairs"] / max(s["draws"], 1),
            "partner_5c_attempts": ps["attempts"], "partner_5c_rejected": ps["rejected"],
            "5c_examples": s["examples"][:5], "partner_5c_examples": ps["examples"][:5]}
    for sp, ids in splits.items():
        with open(f"{root}/splits/{sp}.txt", "w") as f:
            f.write("\n".join(sorted(ids)) + ("\n" if ids else ""))
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()
    except Exception:
        commit = None
    counts = {sp: {} for sp in SPLITS}
    for j in res:
        for p in j["problems"]:
            key = p["problem_id"].rsplit("-", 2)[0]
            counts[p["split"]][key] = counts[p["split"]].get(key, 0) + 1
    man = {"contract_version": CONTRACT_VERSION, "generator_version": GENERATOR_VERSION, "git_commit": commit,
           "rules": {r: {"type": RULE_TYPE[r], "program": RULE_PROGRAM[r], "knob": KNOB_NAME[r],
                         "a_partner": PARTNERS.get(r)} for r in a.rules.split(",")},
           "a_partner_settings": {f"{v}p{k}-s01": {"a_rule": v, "base_b_rule": k} for k, v in PARTNERS.items()},
           "tuning": TUNING, "splits": list(SPLITS), "counts": counts,
           "n_panels": len(tasks), "n_problems": sum(len(v) for v in splits.values()),
           "model_input_sizes": [224, 448], "render_styles": {"default": "solid black strokes and fills",
           "hatch": "filled regions as 45-degree hatching (spacing 0.025 panel width, line = 1.5x stroke) inside a "
                    "solid boundary; all strokes 1.5x the panel stroke width"},
           "not_applicable": {"C1": ["ood_count"]}, "assembly_stats": stats}
    with open(f"{root}/manifest.json", "w") as f:
        json.dump(man, f, indent=1)
    print(json.dumps({k: (v["problems"], round(v["5c_resample_rate"], 3), v["partner_5c_rejected"]) for k, v in stats.items()}))


if __name__ == "__main__":
    main()
