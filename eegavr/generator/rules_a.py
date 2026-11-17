import numpy as np
from shapely.geometry import Polygon, Point
from . import scene as S
from .scene import (make_obj, finalize, poly, ink_area, layout_ok, in_panel, gap_ok, sample_stroke, SHAPES, CONCAVE,
                    SYMMETRY, CHIRAL, SIZE_RANGE, POS_RANGE)

A_RULES = ("A2", "A3", "A4", "A5")
ID_N, OOD_N, SMALL = (2, 4), (7, 8), (0.06, 0.10)
A3_THRESHOLD = 0.11
A4_MIN_TILT = 10.0
INK_TOL = 0.01
NON_TRI = tuple(s for s in SHAPES if s not in ("triangle", "pentagon"))
CONVEX = tuple(s for s in SHAPES if s not in CONCAVE)
CONVEX_SWAP = ("square", "pentagon", "right_trapezoid", "irregular_pentagon")
TILTABLE = tuple(s for s in SHAPES if s != "circle")
MATCH_STEP = 1.0


def _unit_depth(shape):
    # deepest point of the shape's outline below its convex hull, for unit size
    P = Polygon(S.UNIT[shape])
    H = P.convex_hull.exterior
    pts = P.exterior.segmentize(0.01).coords
    return float(max(H.distance(Point(p)) for p in pts))


UNIT_DEPTH = {k: _unit_depth(k) for k in CONCAVE}


def period(o):
    k = o["symmetry_order"]
    return 360.0 / k if k else 360.0


def tilt(o):
    # angular distance of the rotation to the nearest upright pose, modulo the rotational symmetry
    r = o["rotation"] % period(o)
    return min(r, period(o) - r)


def concavity_depth(o):
    return o["size"] * UNIT_DEPTH[o["shape"]] if o["shape"] in CONCAVE else 0.0


def a3(scene, t=A3_THRESHOLD):
    return any(o["size"] >= t for o in scene["objects"])


def a4(scene, t=A4_MIN_TILT):
    return any(o["shape"] != "circle" and tilt(o) >= t for o in scene["objects"])


def a4_near45(scene, tol=5.0):
    return any(o["shape"] != "circle" and abs(((o["rotation"] - 45.0) + period(o) / 2) % period(o) - period(o) / 2) <= tol
               for o in scene["objects"])


def a5(scene, depth=0.0):
    return any(o["shape"] in CONCAVE and concavity_depth(o) >= depth for o in scene["objects"])


PREDICATES = {"A3": a3, "A4": a4, "A5": a5}
FAMILY_A = {"A3": ("A3:",), "A4": ("A4:",), "A5": ("A5:",)}


def candidates():
    # 5c candidates of this track: each rule at several knob thresholds
    C = {}
    for t in (0.08, 0.09, 0.11, 0.13, 0.15):
        C[f"A3:size>={t} present"] = lambda s, t=t: a3(s, t)
    C["A3:largest/smallest size>=1.5"] = lambda s: _size_ratio(s) >= 1.5
    for t in (5.0, 10.0, 15.0, 25.0, 35.0):
        C[f"A4:tilt>={t:g} present"] = lambda s, t=t: a4(s, t)
    C["A4:tilted 45+-5 present"] = a4_near45
    for d in (0.0, 0.03, 0.045, 0.06):
        C[f"A5:concave depth>={d} present"] = lambda s, d=d: a5(s, d)
    return C


def _size_ratio(s):
    z = [o["size"] for o in s["objects"]]
    return max(z) / min(z)


def _u(rng, r):
    return float(rng.uniform(r[0], r[1]))


def _clone(objs):
    return [dict(o) for o in objs]


def _index(objs):
    for i, o in enumerate(objs):
        o["idx"] = i
    return objs


def _place(rng, objs, n, make, tries=60):
    # append n objects from make() with the minimum gap; None if they do not fit
    out = list(objs)
    for _ in range(n):
        for _ in range(tries):
            d = make()
            if in_panel(d) and all(gap_ok(d, o) for o in out):
                out.append(d)
                break
        else:
            return None
    return out


def nn_gap(objs, j):
    q = poly(objs[j])
    return min(q.distance(poly(o)) for i, o in enumerate(objs) if i != j)


def others_min_gap(objs, j):
    ps = [poly(o) for i, o in enumerate(objs) if i != j]
    return min((ps[a].distance(ps[b]) for a in range(len(ps)) for b in range(a + 1, len(ps))), default=np.inf)


def min_gap(objs):
    ps = [poly(o) for o in objs]
    return min(ps[a].distance(ps[b]) for a in range(len(ps)) for b in range(a + 1, len(ps)))


def _not_closest(objs, j):
    return len(objs) >= 3 and nn_gap(objs, j) > others_min_gap(objs, j)


def matched_rotation(rng, objs, j, shape, mirror):
    # rotation of a replacement shape at object j that keeps j's nearest-neighbour polygon gap unchanged
    target = nn_gap(objs, j)
    others = [poly(o) for i, o in enumerate(objs) if i != j]
    per = 360.0 / SYMMETRY[shape] if SYMMETRY[shape] else 360.0

    def f(r):
        q = poly(dict(objs[j], shape=shape, mirror=mirror, rotation=r))
        return min(q.distance(p) for p in others) - target

    grid = np.arange(0.0, per + 1e-9, MATCH_STEP * per / 72.0)
    v = np.array([f(r) for r in grid])
    idx = np.nonzero(np.sign(v[:-1]) != np.sign(v[1:]))[0]
    if len(idx) == 0:
        return None
    i = int(idx[int(rng.integers(len(idx)))])
    lo, hi, flo = grid[i], grid[i + 1], v[i]
    for _ in range(30):
        m = 0.5 * (lo + hi)
        fm = f(m)
        lo, hi, flo = (m, hi, fm) if np.sign(fm) == np.sign(flo) else (lo, m, flo)
    return float((0.5 * (lo + hi)) % per)


def _upright(rng, shape):
    k = SYMMETRY[shape]
    return float(360.0 / k * int(rng.integers(k))) if k else 0.0


def _ink(objs):
    return sum(ink_area(o) for o in objs)


def _swap(objs, j, shape, mirror, rotation):
    out = _clone(objs)
    out[j] = finalize(dict(out[j], shape=shape, mirror=bool(mirror and CHIRAL[shape]), rotation=float(rotation)))
    return out


def _declared(pos, twin, j):
    return {(j, f) for f in ("shape", "rotation", "mirror", "size") if pos[j][f] != twin[j][f]}


def _a2(rng, st):
    # one outline triangle; twin: same-size (same-perimeter) outline pentagon rotated to keep the gap structure
    stroke = sample_stroke(rng, st["style"])
    sr = st["size_range"]
    k = int(rng.integers(st["knob"][0], st["knob"][1] + 1))
    T = make_obj("triangle", _u(rng, sr), False, _u(rng, POS_RANGE), _u(rng, POS_RANGE), rng.uniform(0, 360), stroke,
                 "t1")
    if not in_panel(T):
        return None
    p_pent = 1.0 - 0.5 ** (1.0 / k)

    def mk():
        if rng.random() < p_pent:
            return S.random_obj(rng, ["pentagon"], sr, stroke, fill=False)
        return S.random_obj(rng, list(NON_TRI), sr, stroke)

    objs = _place(rng, [T], k, mk)
    if objs is None:
        return None
    rot = matched_rotation(rng, objs, 0, "pentagon", False)
    if rot is None:
        return None
    pos = _index(_clone(objs))
    twin = _index(_swap(pos, 0, "pentagon", False, rot))
    if not layout_ok(twin):
        return None
    return pos, twin, k, _declared(pos, twin, 0), {"target": 0}


def _a3(rng, st):
    # one shape at threshold + g; twin shrinks it to threshold - g; the target is never in the closest pair
    stroke = sample_stroke(rng, st["style"])
    T, g = st["a3_threshold"], _u(rng, st["knob"])
    lo, hi = st["size_range"][0], min(st["size_range"][1], T - g)
    if hi <= lo:
        return None
    L = S.random_obj(rng, SHAPES, (T + g, T + g), stroke, role="t1")
    if not in_panel(L):
        return None
    objs = _place(rng, [L], st["n_objects"][0] - 1, lambda: S.random_obj(rng, SHAPES, (lo, hi), stroke))
    if objs is None or not _not_closest(objs, 0):
        return None
    pos = _index(_clone(objs))
    twin = _clone(pos)
    twin[0] = finalize(dict(twin[0], size=T - g))
    if not (_not_closest(twin, 0) and abs(min_gap(twin) - min_gap(pos)) < 1e-12):
        return None
    return pos, twin, g, {(0, "size")}, {"target": 0}


def _a4(rng, st):
    # one non-circular shape tilted 45 deg - knob (modulo symmetry); every other shape upright; twin: upright
    stroke = sample_stroke(rng, st["style"])
    sr = st["size_range"]
    d = _u(rng, st["knob"])
    t = 45.0 - d
    feas = [s for s in TILTABLE if (360.0 / SYMMETRY[s]) / 2 >= t]
    sh = feas[int(rng.integers(len(feas)))]
    base = _upright(rng, sh)
    sign = 1.0 if rng.random() < 0.5 else -1.0
    A = make_obj(sh, _u(rng, sr), bool(rng.random() < 0.5), _u(rng, POS_RANGE), _u(rng, POS_RANGE), base + sign * t,
                 stroke, "t1", mirror=bool(rng.random() < 0.5))
    if not in_panel(A):
        return None

    def mk():
        o = S.random_obj(rng, SHAPES, sr, stroke)
        return finalize(dict(o, rotation=_upright(rng, o["shape"])))

    objs = _place(rng, [A], st["n_objects"][0] - 1, mk)
    if objs is None or not _not_closest(objs, 0):
        return None
    pos = _index(_clone(objs))
    twin = _clone(pos)
    twin[0] = finalize(dict(twin[0], rotation=base))
    if not (in_panel(twin[0]) and layout_ok(twin) and _not_closest(twin, 0)
            and abs(min_gap(twin) - min_gap(pos)) < 1e-12):
        return None
    return pos, twin, d, {(0, "rotation")}, {"target": 0, "tilt": t}


def _a5(rng, st):
    # one outline cross or L with the knob's concavity depth; twin: same-size convex shape rotated to keep the gaps
    stroke = sample_stroke(rng, st["style"])
    sr = st["size_range"]
    d = _u(rng, st["knob"])
    feas = [s for s in CONCAVE if sr[0] <= d / UNIT_DEPTH[s] <= sr[1]]
    if not feas:
        return None
    sh = feas[int(rng.integers(len(feas)))]
    A = make_obj(sh, d / UNIT_DEPTH[sh], False, _u(rng, POS_RANGE), _u(rng, POS_RANGE), rng.uniform(0, 360), stroke,
                 "t1", mirror=bool(rng.random() < 0.5))
    if not in_panel(A):
        return None
    objs = _place(rng, [A], st["n_objects"][0] - 1, lambda: S.random_obj(rng, list(CONVEX), sr, stroke))
    if objs is None:
        return None
    new = CONVEX_SWAP[int(rng.integers(len(CONVEX_SWAP)))]
    mir = bool(rng.random() < 0.5) and CHIRAL[new]
    rot = matched_rotation(rng, objs, 0, new, mir)
    if rot is None:
        return None
    pos = _index(_clone(objs))
    twin = _index(_swap(pos, 0, new, mir, rot))
    if not layout_ok(twin) or abs(_ink(twin) - _ink(pos)) / _ink(pos) > INK_TOL:
        return None
    return pos, twin, d, _declared(pos, twin, 0), {"target": 0}


SAMPLERS = {"A2": _a2, "A3": _a3, "A4": _a4, "A5": _a5}


def rule_diff(pair, size=224):
    # rule-relevant difference in px at one input size: Hausdorff distance of the target outline (A5: also the depth)
    j = pair["meta"]["target"]
    a, b = pair["pos"]["objects"][j], pair["twin"]["objects"][j]
    h = poly(a).hausdorff_distance(poly(b)) * size
    return min(h, concavity_depth(a) * size) if a["shape"] in CONCAVE else h


KNOB_NAME_A = {"A2": "number of non-triangle shapes",
               "A3": "size gap g to the 0.11 threshold (positive 0.11 + g, twin 0.11 - g)",
               "A4": "angle distance of the target's tilt from 45 deg (twin: upright)",
               "A5": "concavity depth of the cross or L (panel widths, set through its size)"}
RULE_PROGRAM_A = {
    "A2": "exists o: o.shape == triangle",
    "A3": "exists o: o.size >= 0.11  (perimeter-equivalent radius; all other shapes below 0.11 - g)",
    "A4": "exists o: o.shape != circle and tilt(o) >= 10 deg, tilt = angular distance of o.rotation to the nearest "
          "multiple of 360/symmetry_order (every other non-circular shape is upright)",
    "A5": "exists o: o.shape in {cross, L}  (concave)",
}
ID_BINS_A = {
    "A2": [(1, 1), (2, 2), (3, 3)],
    "A3": [(0.010, 0.020), (0.020, 0.030), (0.030, 0.045)],
    "A4": [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)],
    "A5": [(0.022, 0.035), (0.035, 0.050), (0.050, 0.090)],
}
FULL_A = {"A2": (1, 3), "A3": (0.010, 0.045), "A4": (0.0, 30.0), "A5": (0.022, 0.090)}
OOD_KNOB_A = {"A2": (6, 7), "A3": (0.010, 0.045), "A4": (0.0, 30.0), "A5": (0.022, 0.056)}
N_ID_A = {"A2": None, "A3": (3, 4), "A4": (3, 4), "A5": ID_N}
TUNING_A = {"A3": {"a3_threshold": A3_THRESHOLD}}


def _st(rule, sid, knob, n, size_range, style="default"):
    st = {"id": sid, "rule": rule, "type": "A", "knob": tuple(knob), "knob_name": KNOB_NAME_A[rule],
          "n_objects": None if n is None else tuple(n), "size_range": tuple(size_range), "style": style}
    st.update(TUNING_A.get(rule, {}))
    return st


def id_settings_a(rule):
    return [_st(rule, f"{rule}-s{i + 1:02d}", b, N_ID_A[rule], SIZE_RANGE) for i, b in enumerate(ID_BINS_A[rule])]


def ood_settings_a(rule):
    # oc: 7-8 objects, sizes capped at 0.10 (A3: the target excepted); occ: ID count, same cap; or: hatch style
    full, n_id = FULL_A[rule], N_ID_A[rule]
    return {"ood_count": _st(rule, f"{rule}-oc", OOD_KNOB_A[rule], None if rule == "A2" else OOD_N, SMALL),
            "ood_count_control": _st(rule, f"{rule}-occ", OOD_KNOB_A[rule] if rule == "A5" else full, n_id, SMALL),
            "ood_render": _st(rule, f"{rule}-or", full, n_id, SIZE_RANGE, "hatch")}


def register_settings(ns):
    # adds this track's rule metadata and settings to the settings-module namespace ns
    ns["RULE_TYPE"].update({r: "A" for r in A_RULES})
    ns["RULE_PROGRAM"].update(RULE_PROGRAM_A)
    ns["KNOB_NAME"].update(KNOB_NAME_A)
    ns["EXTRA_ID"].update({r: id_settings_a(r) for r in A_RULES})
    ns["EXTRA_OOD"].update({r: ood_settings_a(r) for r in A_RULES})
    ns["V20_RULES"].extend(A_RULES)
