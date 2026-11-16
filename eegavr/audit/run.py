import json
import os
import time
from multiprocessing import Pool
import numpy as np
from ..generator.settings import all_settings, seed_for, b_setting_for, SETTING_CODE
from ..generator.rules import sample_pair
from ..generator.partners import partner_pair
from ..generator.render import render, render_mask
from ..generator.version import GENERATOR_VERSION
from ..generator.scene import ink_area
from .features import pixel_features, pass_columns, global_columns, ALL_NAMES, ROLE_ABS_NAMES
from .leakage import symbolic_features, c_cues, C_CUES
from .rendering import check_panel
from .competing import literal_matrix, equivalence_mask
from .probes import run_probes, run_univariate

SIZES = (224, 448)


def make_pair(st, i, split="audit"):
    # the i-th pair of a setting (partner settings derive from the B setting's pairs)
    if "partner_of" in st:
        bst = b_setting_for(st["partner_of"], i)
        bp = sample_pair(bst, seed_for(split, bst["id"], i))
        return partner_pair(st["rule"], bp, state=(i % 2 == 0))
    ctrl = st.get("c1_controls", ["equal_size", "equal_ink"])[i % 2] if st["rule"] == "C1" else None
    p = sample_pair(st, seed_for(split, st["id"], i), control=ctrl)
    p["meta"]["control"] = ctrl
    return p


def _work(args):
    sid, i, tuning = args
    st = all_settings()[sid]
    st.update(tuning)
    p = make_pair(st, i)
    rows = []
    for lab, sc in ((1, p["pos"]), (0, p["twin"])):
        img = render(sc)
        m = render_mask(sc)
        d = {s: check_panel(img, sc, s) for s in SIZES}
        rows.append({"y": lab, "pair": i, "pix": pixel_features(img, m, sc),
                     "sym": symbolic_features(sc, st["type"]), "d5": d, "ink": sum(ink_area(o) for o in sc["objects"]),
                     "lit": literal_matrix([sc])[0][:, 0], "nobj": len(sc["objects"]),
                     "cues": c_cues(sc) if st["type"] == "C" else None})
    return rows, p["draws"], p["knob"], p.get("meta", {}).get("control")


def audit_setting(sid, n_pairs=1000, workers=8, tuning=None, out_dir="runs/audit"):
    # full 5a/5b/5d audit of one rule setting plus the 5c reference pool; writes <sid>.json and pool npz
    t0 = time.time()
    st = all_settings()[sid]
    st.update(tuning or {})
    with Pool(workers) as pool:
        res = pool.map(_work, [(sid, i, tuning or {}) for i in range(n_pairs)], chunksize=8)
    rows = [r for rr, *_ in res for r in rr]
    draws = sum(r[1] for r in res)
    y = np.array([r["y"] for r in rows])
    g = np.array([r["pair"] for r in rows])
    P = np.array([r["pix"] for r in rows])
    full_cols = [k for k, n in enumerate(ALL_NAMES) if n not in ROLE_ABS_NAMES]
    a5 = {"pass_set": run_probes(P[:, pass_columns(st["rule"])], y, g),
          "full_set": run_probes(P[:, full_cols], y, g),
          "global_only": run_probes(P[:, global_columns()], y, g),
          "diagnostic_with_abs_role_centroids": run_probes(P, y, g)}
    b5 = run_probes(np.array([r["sym"] for r in rows]), y, g)
    d5 = {}
    for s in SIZES:
        ch = [r["d5"][s] for r in rows]
        d5[s] = {k: float(np.mean([c[k] for c in ch])) for k in ("no_clip", "components_ok", "gap_ok", "stroke_ok")}
        d5[s]["min_gap_px"] = float(min(c["min_gap_px"] for c in ch))
        d5[s]["all_ok"] = float(np.mean([all(c[k] for k in ("no_clip", "components_ok", "gap_ok", "stroke_ok"))
                                         for c in ch]))
    L = np.array([r["lit"] for r in rows]).T
    eq = equivalence_mask(L, y.astype(bool))
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(f"{out_dir}/{sid}_pool.npz", eq=eq, pix=P, sym=np.array([r["sym"] for r in rows]), y=y, g=g)
    ink = np.array([r["ink"] for r in rows])
    rtype = st["type"]
    uni = None
    if rtype == "C":
        gc = global_columns()
        uni = {"pixel": run_univariate(P[:, gc], y, g, [ALL_NAMES[k] for k in gc]),
               "symbolic": run_univariate(np.array([r["cues"] for r in rows]), y, g, C_CUES)}
    p5a = a5["pass_set"]["pass"] if rtype == "B" else (uni["pixel"]["pass"] if rtype == "C" else None)
    p5d = bool(d5[224]["all_ok"] == 1.0)
    p5b = uni["symbolic"]["pass"] if rtype == "C" else b5["pass"]
    passed = bool(p5b and p5d and (p5a if p5a is not None else True))
    knobs = [r[2] for r in res]
    rep = {"setting_id": sid, "rule": st["rule"], "type": rtype, "generator_version": GENERATOR_VERSION,
           "setting": {k: (list(v) if isinstance(v, tuple) else v) for k, v in st.items()},
           "n_pairs": n_pairs, "n_panels": len(rows), "kept_rate": n_pairs / draws, "draws": draws,
           "knob_mean": float(np.mean(knobs)), "knob_min": float(np.min(knobs)), "knob_max": float(np.max(knobs)),
           "ink_ratio_twin_over_pos_mean": float(np.mean(ink[y == 0] / ink[y == 1])),
           "5a": a5, "5a_criterion": "pass_set CI90 inside [0.45,0.55]" if p5a is not None else "record only (Type A)",
           "5b": b5, "5d": d5, "pass_5a": p5a, "pass_5b": p5b, "pass_5d": p5d, "pass": passed,
           "c_univariate": uni,
           "criterion": {"A": "5a record only; 5b relation-only probe CI90 in band; 5d all panels ok",
                         "B": "5a pass probe = role-free global pixel features, CI90 in [0.45,0.55]; 5b positions "
                              "hidden, CI90 in band; 5d all panels ok",
                         "C": "univariate probes on each single global pixel feature (5a) and on each numerosity cue "
                              "(5b: total ink, mean item size, field hull, density) CI90 in band; joint probes "
                              "reported, not a criterion; 5d all panels ok"}[rtype],
           "seconds": time.time() - t0}
    with open(f"{out_dir}/{sid}.json", "w") as f:
        json.dump(rep, f, indent=1)
    with open(f"{out_dir}/audit_log.jsonl", "a") as f:
        f.write(json.dumps({"setting_id": sid, "generator_version": GENERATOR_VERSION, "tuning": tuning or {},
                            "n_pairs": n_pairs, "kept_rate": rep["kept_rate"],
                            "5a_pass_set": [a5["pass_set"]["best"], a5["pass_set"]["best_acc"],
                                            a5["pass_set"]["best_ci90"]],
                            "5a_full_set": [a5["full_set"]["best"], a5["full_set"]["best_acc"],
                                            a5["full_set"]["best_ci90"]],
                            "5b": [b5["best"], b5["best_acc"], b5["best_ci90"]], "5d_224_all_ok": d5[224]["all_ok"],
                            "c_univariate_worst": None if uni is None else
                            {k: [v["worst_cue"], v["worst_acc"], v["worst_ci90"]] for k, v in uni.items()},
                            "5d_448_all_ok": d5[448]["all_ok"], "pass": passed, "time": time.strftime("%F %T")}) + "\n")
    return rep
