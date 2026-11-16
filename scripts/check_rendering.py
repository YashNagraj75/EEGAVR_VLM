import argparse
import json
import os
import sys
from multiprocessing import Pool
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eegavr.audit.rendering import check_panel

SIZES = (224, 256, 448)
KEYS = ("no_clip", "components_ok", "gap_ok", "stroke_ok")


def _one(args):
    root, pid = args
    sc = json.load(open(f"{root}/scenes/{pid}.json"))
    img = np.asarray(Image.open(f"{root}/panels/{pid}.png"))
    return pid, {s: check_panel(img, sc, s) for s in SIZES}


def main():
    # per-panel 5d over an exported set (BOX resampling, 224/256/448 px); writes audit/rendering_check.json
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    pids = [f[:-5] for f in os.listdir(f"{a.root}/scenes")]
    with Pool(a.workers) as pool:
        res = dict(pool.imap_unordered(_one, [(a.root, p) for p in pids], chunksize=64))
    prob_fail = {}
    for f in os.listdir(f"{a.root}/problems"):
        p = json.load(open(f"{a.root}/problems/{f}"))
        ids = p["left"] + p["right"] + [t["panel_id"] for t in p["test"]]
        bad = [i for i in ids for s in SIZES if not all(res[i][s][k] for k in KEYS)]
        if bad:
            prob_fail[p["problem_id"]] = sorted(set(bad))
    summ = {}
    for s in SIZES:
        summ[s] = {k: float(np.mean([r[s][k] for r in res.values()])) for k in KEYS}
        summ[s]["min_gap_px"] = float(min(r[s]["min_gap_px"] for r in res.values()))
        summ[s]["panels_failing"] = int(sum(not all(r[s][k] for k in KEYS) for r in res.values()))
    out = {"resample": "PIL BOX", "sizes": SIZES, "n_panels": len(res), "summary": summ,
           "problems_failing": prob_fail, "n_problems_failing": len(prob_fail)}
    json.dump(out, open(f"{a.root}/audit/rendering_check.json", "w"), indent=1)
    print(json.dumps({"n_panels": len(res), "n_problems_failing": len(prob_fail),
                      "summary": {s: summ[s]["panels_failing"] for s in SIZES},
                      "min_gap_px": {s: round(summ[s]["min_gap_px"], 2) for s in SIZES}}))


if __name__ == "__main__":
    main()
