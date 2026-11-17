from itertools import combinations
import numpy as np
from .scene import (make_obj, moved, poly, layout_ok, in_panel, gap_ok, index, sample_stroke, random_obj, ink_area,
                    SHAPES, EDGE_MARGIN, POS_RANGE)

RULES = ("B2", "B5", "B8", "B9")
ON_TOL = 0.004
OTHER_MIN = 0.02
MIN_RULE_PX = 2.0
SMALLEST_PX = 224
B2_DISTRACTOR_SHAPES = [s for s in SHAPES if s not in ("triangle", "circle")]
CONFUSABLE = {"pentagon": "irregular_pentagon", "irregular_pentagon": "pentagon"}


def _u(rng, r):
    return float(rng.uniform(r[0], r[1]))


def _n(rng, st):
    lo, hi = st["n_objects"]
    return int(rng.integers(lo, hi + 1))


def _clone(objs):
    return [dict(o) for o in objs]


def _xy(o):
    return np.array([o["x"], o["y"]])


def triple_dev(a, b, c):
    # height of the centroid triangle over its longest side, and the position (0-2) of the middle object
    P = [_xy(a), _xy(b), _xy(c)]
    sides = [np.hypot(*(P[1] - P[2])), np.hypot(*(P[0] - P[2])), np.hypot(*(P[0] - P[1]))]
    k = int(np.argmax(sides))
    area2 = abs((P[1][0] - P[0][0]) * (P[2][1] - P[0][1]) - (P[1][1] - P[0][1]) * (P[2][0] - P[0][0]))
    return float(area2 / max(sides[k], 1e-12)), k


def left_of(scene, gap=0.0, a="triangle", b="circle"):
    ob = scene["objects"]
    return any(q["bbox"][0] - p["bbox"][2] > gap for p in ob if p["shape"] == a for q in ob if q["shape"] == b)


def b5(scene, tol=ON_TOL):
    return any(triple_dev(*t)[0] <= tol for t in combinations(scene["objects"], 3))


def _between(scene, tol, odd_middle):
    # collinear triple whose two ends share a shape and whose middle differs (odd_middle) or matches one end
    for t in combinations(scene["objects"], 3):
        d, k = triple_dev(*t)
        if d > tol:
            continue
        m, e = t[k], [t[i] for i in range(3) if i != k]
        if odd_middle and e[0]["shape"] == e[1]["shape"] and m["shape"] != e[0]["shape"]:
            return True
        if not odd_middle and e[0]["shape"] != e[1]["shape"] and m["shape"] in (e[0]["shape"], e[1]["shape"]):
            return True
    return False


def b8(scene, tol=ON_TOL):
    return _between(scene, tol, True)


def _pairs(ob):
    return sorted((float(np.hypot(*(_xy(a) - _xy(b)))), i, j) for (i, a), (j, b) in combinations(enumerate(ob), 2))


def b9(scene, ratio=1.0, same=True):
    # the closest pair (centroid distance) is same-shaped (same=False: differently shaped), by a factor >= ratio
    ob = scene["objects"]
    d = _pairs(ob)
    if len(d) < 2:
        return False
    (d0, i, j), d1 = d[0], d[1][0]
    ok = d1 > d0 if ratio <= 1.0 else d1 >= ratio * d0
    return ok and (ob[i]["shape"] == ob[j]["shape"]) == same


PREDICATES = {"B2": left_of, "B5": b5, "B8": b8, "B9": b9}
FAMILY = {"B2": ("B2:", "B2':"), "B5": ("B5:",), "B8": ("B8:", "B8':"), "B9": ("B9:", "B9':")}


def candidates():
    # B-layout predicates for the 5c check at the knob bin edges, plus their reversed-role variants
    C = {}
    for t in (0.0, 0.03, 0.05, 0.08, 0.12, 0.16, 0.20):
        C[f"B2:triangle left of circle gap>{t}"] = lambda s, t=t: left_of(s, t)
        C[f"B2':circle left of triangle gap>{t}"] = lambda s, t=t: left_of(s, t, "circle", "triangle")
    for t in (ON_TOL, 0.015, 0.03, 0.05):
        C[f"B5:collinear dev<={t}"] = lambda s, t=t: b5(s, t)
    C[f"B8:odd between same dev<={ON_TOL}"] = lambda s: _between(s, ON_TOL, True)
    C[f"B8':same-shaped between dev<={ON_TOL}"] = lambda s: _between(s, ON_TOL, False)
    for r in (1.0, 1.25, 1.5, 2.0):
        C[f"B9:closest pair same-shaped ratio>={r}"] = lambda s, r=r: b9(s, r)
        C[f"B9':closest pair different shapes ratio>={r}"] = lambda s, r=r: b9(s, r, same=False)
    return C


def _rule_px_ok(diff):
    return diff * SMALLEST_PX >= MIN_RULE_PX


def _add_free(rng, panels, n, shapes, size_range, stroke, line_thr=0.0, min_dist=0.0, unique=False, tries=80):
    # shared distractors keeping the gap, the centroid distance and no near-collinear triple in every panel
    added, pool = [], list(shapes)
    for _ in range(n):
        for _ in range(tries):
            if not pool:
                return None
            d = random_obj(rng, pool, size_range, stroke)
            if not in_panel(d):
                continue
            ok = True
            for P in panels:
                rest = P + added
                if not all(gap_ok(d, o) and np.hypot(*(_xy(d) - _xy(o))) >= min_dist for o in rest):
                    ok = False
                    break
                if line_thr > 0 and any(triple_dev(d, a, b)[0] < line_thr for a, b in combinations(rest, 2)):
                    ok = False
                    break
            if ok:
                added.append(d)
                if unique:
                    pool = [x for x in pool if x not in (d["shape"], CONFUSABLE.get(d["shape"]))]
                break
        else:
            return None
    return added


def _b2(rng, st):
    # twin swaps the two roles' ink-centroid x (y kept); extent predicate checked on both panels
    from .rules import equal_ink_partner
    from .scene import add_distractors
    stroke = sample_stroke(rng, st["style"])
    fill = bool(rng.random() < 0.5) if st.get("role_fill") != "outline" else False
    sz = equal_ink_partner(rng, "circle", "triangle", st["size_range"], fill, stroke,
                           st.get("role_ink_jitter", 0.0), st.get("role_ink_offset", 0.0))
    if sz is None:
        return None
    sc, stri = sz
    T = make_obj("triangle", stri, fill, .5, _u(rng, POS_RANGE), rng.uniform(0, 360), stroke, "t1")
    C = make_obj("circle", sc, fill, .5, _u(rng, POS_RANGE), 0, stroke, "t2")
    tl, tr = T["x"] - T["bbox"][0], T["bbox"][2] - T["x"]
    cl, cr = C["x"] - C["bbox"][0], C["bbox"][2] - C["x"]
    g = _u(rng, st["knob"])
    W = tl + tr + g + cl + cr
    if W > 1 - 2 * EDGE_MARGIN:
        return None
    left = rng.uniform(EDGE_MARGIN, 1 - EDGE_MARGIN - W)
    xT, xC = left + tl, left + tl + tr + g + cl
    pos = [moved(T, x=xT), moved(C, x=xC)]
    twn = [moved(T, x=xC), moved(C, x=xT)]
    if not (layout_ok(pos) and layout_ok(twn)):
        return None
    if not twn[0]["bbox"][0] > twn[1]["bbox"][2]:
        return None
    objs = add_distractors(rng, pos, _n(rng, st) - 2, B2_DISTRACTOR_SHAPES, st["size_range"], stroke)
    if objs is None:
        return None
    pos = index(_clone(objs))
    twin = index(twn + _clone(objs[2:]))
    if not layout_ok(twin):
        return None
    return pos, twin, pos[1]["bbox"][0] - pos[0]["bbox"][2], {(0, "x"), (1, "x")}, {}


def _b5(rng, st):
    # three random shapes with collinear centroids; twin moves the middle one off the line by d
    stroke = sample_stroke(rng, st["style"])
    sr = st["size_range"]
    mid, e1, e2 = [random_obj(rng, SHAPES, sr, stroke, role=r) for r in ("t1", "t2", "t2")]
    phi = rng.uniform(0, np.pi)
    u, nv = np.array([np.cos(phi), np.sin(phi)]), np.array([-np.sin(phi), np.cos(phi)])
    gr = st.get("b5_gap", (0.03, 0.15))
    s1, s2 = mid["radius"] + e1["radius"] + _u(rng, gr), mid["radius"] + e2["radius"] + _u(rng, gr)
    M = np.array([_u(rng, POS_RANGE), _u(rng, POS_RANGE)])
    P = [M, M - s1 * u, M + s2 * u]
    d = _u(rng, st["knob"])
    if not _rule_px_ok(d):
        return None
    sgn = 1.0 if rng.random() < 0.5 else -1.0
    a, b = d, 0.0
    if st.get("b5_balanced"):
        w = [ink_area(o) for o in (mid, e1, e2)]
        b = -w[0] * d / sum(w)
        a = d + b
    shift = [a * sgn * nv, b * sgn * nv, b * sgn * nv]
    pos = [moved(o, x=p[0], y=p[1]) for o, p in zip((mid, e1, e2), P)]
    twn = [moved(o, x=p[0] + q[0], y=p[1] + q[1]) for o, p, q in zip((mid, e1, e2), P, shift)]
    if not (layout_ok(pos) and layout_ok(twn)):
        return None
    if triple_dev(*twn)[0] < st["knob"][0] - 1e-9:
        return None
    thr = max(OTHER_MIN, st["knob"][0])
    ds = _add_free(rng, [pos, twn], _n(rng, st) - 3, SHAPES, sr, stroke, line_thr=thr)
    if ds is None:
        return None
    dec = {(0, "x"), (0, "y")} | ({(i, f) for i in (1, 2) for f in ("x", "y")} if st.get("b5_balanced") else set())
    return index(_clone(pos + ds)), index(_clone(twn + ds)), triple_dev(*twn)[0], dec, {}


def _sop_roles(rng, st, stroke):
    # odd shape O (t1) and a same-shaped pair S, S (t2), never a confusable pair, equal ink (+- jitter), shared fill
    from .rules import equal_ink_partner
    k = rng.permutation(len(SHAPES))[:2]
    S, O = SHAPES[k[0]], SHAPES[k[1]]
    if CONFUSABLE.get(S) == O:
        return None
    fill = bool(rng.random() < 0.5) if st.get("role_fill") != "outline" else False
    sz = equal_ink_partner(rng, S, O, st["size_range"], fill, stroke, st.get("role_ink_jitter", 0.0))
    if sz is None:
        return None
    mir = bool(rng.random() < 0.5)
    o = make_obj(O, sz[1], fill, .5, .5, rng.uniform(0, 360), stroke, "t1", mirror=bool(rng.random() < 0.5))
    s1 = make_obj(S, sz[0], fill, .5, .5, rng.uniform(0, 360), stroke, "t2", mirror=mir)
    s2 = make_obj(S, sz[0], fill, .5, .5, rng.uniform(0, 360), stroke, "t2", mirror=mir)
    return o, s1, s2


def _swap(objs, i, j):
    out = _clone(objs)
    out[i] = moved(objs[i], x=objs[j]["x"], y=objs[j]["y"])
    out[j] = moved(objs[j], x=objs[i]["x"], y=objs[i]["y"])
    return out


def _b8(rng, st):
    # S O S on a line at equal centre spacing; twin swaps O with a random end shape (positions unchanged as a set)
    stroke = sample_stroke(rng, st["style"])
    r = _sop_roles(rng, st, stroke)
    if r is None:
        return None
    o, s1, s2 = r
    g = _u(rng, st["knob"])
    s = o["radius"] + s1["radius"] + g
    phi = rng.uniform(0, np.pi)
    u = np.array([np.cos(phi), np.sin(phi)])
    M = np.array([_u(rng, POS_RANGE), _u(rng, POS_RANGE)])
    roles = [moved(o, x=M[0], y=M[1]), moved(s1, x=M[0] - s * u[0], y=M[1] - s * u[1]),
             moved(s2, x=M[0] + s * u[0], y=M[1] + s * u[1])]
    j = 1 + int(rng.integers(2))
    twn = _swap(roles, 0, j)
    if not (layout_ok(roles) and layout_ok(twn)):
        return None
    used = {o["shape"], s1["shape"]} | {CONFUSABLE.get(o["shape"]), CONFUSABLE.get(s1["shape"])}
    ds = _add_free(rng, [roles, twn], _n(rng, st) - 3, [x for x in SHAPES if x not in used], st["size_range"],
                   stroke, line_thr=OTHER_MIN, unique=True)
    if ds is None:
        return None
    pos, twin = index(_clone(roles + ds)), index(_clone(twn + ds))
    if not (layout_ok(pos) and layout_ok(twin)):
        return None
    return pos, twin, g, {(0, "x"), (0, "y"), (j, "x"), (j, "y")}, {}


def _closest_by_gap(objs):
    ps = [poly(o) for o in objs]
    return min((ps[i].distance(ps[j]), i, j) for i, j in combinations(range(len(objs)), 2))[1:]


def _b9(rng, st):
    # S-S centroid distance d1 is the smallest, every other pair >= r*d1; twin swaps O with one S
    stroke = sample_stroke(rng, st["style"])
    rr = _sop_roles(rng, st, stroke)
    if rr is None:
        return None
    o, s1, s2 = rr
    r = _u(rng, st["knob"])
    d1 = 2 * s1["radius"] + _u(rng, st.get("b9_gap", (0.03, 0.10)))
    pa = np.array([_u(rng, POS_RANGE), _u(rng, POS_RANGE)])
    t1, t2 = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
    pb = pa + d1 * np.array([np.cos(t1), np.sin(t1)])
    near, far = (pa, pb) if rng.random() < 0.5 else (pb, pa)
    pc = near + r * d1 * np.array([np.cos(t2), np.sin(t2)])
    if np.hypot(*(pc - far)) < r * d1:
        return None
    roles = [moved(o, x=pc[0], y=pc[1]), moved(s1, x=pa[0], y=pa[1]), moved(s2, x=pb[0], y=pb[1])]
    j = 1 + int(rng.integers(2))
    twn = _swap(roles, 0, j)
    if not (layout_ok(roles) and layout_ok(twn)):
        return None
    used = {o["shape"], s1["shape"]} | {CONFUSABLE.get(o["shape"]), CONFUSABLE.get(s1["shape"])}
    ds = _add_free(rng, [roles, twn], _n(rng, st) - 3, [x for x in SHAPES if x not in used], st["size_range"],
                   stroke, min_dist=r * d1 * (1 + 1e-6), unique=True)
    if ds is None:
        return None
    pos, twin = index(_clone(roles + ds)), index(_clone(twn + ds))
    if not (layout_ok(pos) and layout_ok(twin)):
        return None
    if _closest_by_gap(pos) != (1, 2) or _closest_by_gap(twin) != tuple(sorted((0, 3 - j))):
        return None
    d = _pairs(pos)
    if not _rule_px_ok(d[1][0] - d[0][0]):
        return None
    return pos, twin, d[1][0] / d[0][0], {(0, "x"), (0, "y"), (j, "x"), (j, "y")}, {}


SAMPLERS = {"B2": _b2, "B5": _b5, "B8": _b8, "B9": _b9}
RULE_TYPE = {r: "B" for r in RULES}
RULE_PROGRAM = {
    "B2": "exists t(triangle), c(circle): c.extent_left - t.extent_right > 0",
    "B5": "exists a, b, c: centroids collinear (height over the longest side <= 0.004); which shape is in the middle is random",
    "B8": "exists a, b, c on a line (<= 0.004): a.shape == b.shape != c.shape and c between a and b",
    "B9": "the closest pair of objects (centroid distance) has the same shape",
}
KNOB_NAME = {"B2": "horizontal gap between the extents", "B5": "deviation of the middle shape from the line in the twin",
             "B8": "gap between neighbouring shapes (centre spacing minus both circumradii)",
             "B9": "ratio of the second-smallest to the smallest pairwise centroid distance"}
ID_BINS = {"B2": [(0.03, 0.06), (0.06, 0.12), (0.12, 0.24)],
           "B5": [(0.015, 0.03), (0.03, 0.05), (0.05, 0.08)],
           "B8": [(0.03, 0.06), (0.06, 0.10), (0.10, 0.16)],
           "B9": [(1.25, 1.5), (1.5, 2.0), (2.0, 3.0)]}
FULL = {"B2": (0.03, 0.24), "B5": (0.015, 0.08), "B8": (0.03, 0.16), "B9": (1.25, 3.0)}
TUNING = {"B2": {"role_fill": "outline", "role_ink_jitter": 0.03, "role_ink_offset": 0.005, "n_objects_id": (4, 4)},
          "B5": {"n_objects_id": (4, 4)},
          "B8": {"role_ink_jitter": 0.03, "n_objects_id": (4, 4)},
          "B9": {"role_ink_jitter": 0.03, "n_objects_id": (4, 4)}}
