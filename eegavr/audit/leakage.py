import numpy as np
from scipy.spatial import ConvexHull
from ..generator.scene import SHAPES, ink_area
from ..generator.predicates import nest_pairs, touch_pairs

B_NAMES = ([f"n_{s}" for s in SHAPES] + ["n_filled", "n_objects", "t1_size", "t2_size", "size_mean", "size_sd",
                                         "size_min", "size_max", "t1_rot_c", "t1_rot_s", "t2_rot_c", "t2_rot_s",
                                         "total_ink"])
ARR_NAMES = ["mean_x", "mean_y", "sd_x", "sd_y", "hull_area", "pair_min", "pair_mean", "pair_max", "span_w",
             "span_h"]
A_NAMES = ["n_objects", "t1_x", "t1_y", "t1_from_centre", "min_gap", "any_inside", "any_touch",
           "collinearity"] + ARR_NAMES
C_NAMES = ["total_ink", "size_mean", "size_sd", "size_min", "size_max", "n_filled", "rot_c", "rot_s"] + ARR_NAMES


def _rot(o):
    k = o["symmetry_order"]
    if k == 0:
        return 0.0, 0.0
    t = np.deg2rad(o["rotation"] * k)
    return float(np.cos(t)), float(np.sin(t))


def _arr(ob):
    P = np.array([[o["x"], o["y"]] for o in ob])
    d = np.hypot(*(P[:, None] - P[None]).transpose(2, 0, 1))[np.triu_indices(len(P), 1)]
    try:
        hull = ConvexHull(P).volume if len(P) >= 3 else 0.0
    except Exception:
        hull = 0.0
    return [P[:, 0].mean(), P[:, 1].mean(), P[:, 0].std(), P[:, 1].std(), hull, d.min(), d.mean(), d.max(),
            np.ptp(P[:, 0]), np.ptp(P[:, 1])]


def _role(ob, r):
    x = [o for o in ob if o["role"] == r]
    return x[0] if x else None


def b_features(sc):
    # Type B: everything except positions
    ob = sc["objects"]
    sz = np.array([o["size"] for o in ob])
    t1, t2 = _role(ob, "t1"), _role(ob, "t2")
    return ([sum(o["shape"] == s for o in ob) for s in SHAPES] +
            [sum(o["fill"] for o in ob), len(ob), t1["size"] if t1 else 0, t2["size"] if t2 else 0, sz.mean(),
             sz.std(), sz.min(), sz.max(), *(_rot(t1) if t1 else (0, 0)), *(_rot(t2) if t2 else (0, 0)),
             sum(ink_area(o) for o in ob)])


def a_features(sc):
    # Type A: arrangement only (offsets, distances, containment, touching, collinearity)
    from ..generator.scene import poly
    ob = sc["objects"]
    t1 = _role(ob, "t1") or ob[0]
    ps = [poly(o) for o in ob]
    gaps = [ps[i].distance(ps[j]) for i in range(len(ps)) for j in range(i + 1, len(ps))]
    P = np.array([[o["x"], o["y"]] for o in ob])
    col = 1.0
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            for k in range(j + 1, len(P)):
                a = abs(np.cross(P[j] - P[i], P[k] - P[i])) / (np.linalg.norm(P[k] - P[i]) + 1e-9)
                col = min(col, a)
    return ([len(ob), t1["x"], t1["y"], np.hypot(t1["x"] - .5, t1["y"] - .5), min(gaps),
             float(bool(nest_pairs(sc))), float(bool(touch_pairs(sc))), col] + _arr(ob))


def c_features(sc):
    # Type C: sizes, ink, fill count, rotations and arrangement, never the count
    ob = sc["objects"]
    sz = np.array([o["size"] for o in ob])
    r = np.array([_rot(o) for o in ob])
    return ([sum(ink_area(o) for o in ob), sz.mean(), sz.std(), sz.min(), sz.max(), sum(o["fill"] for o in ob),
             r[:, 0].mean(), r[:, 1].mean()] + _arr(ob))


def symbolic_features(sc, rtype):
    return np.array({"A": a_features, "B": b_features, "C": c_features}[rtype](sc), float)


def names(rtype):
    return {"A": A_NAMES, "B": B_NAMES, "C": C_NAMES}[rtype]
