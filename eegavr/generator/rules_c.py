from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from .scene import (make_obj, finalize, poly, ink_geom, in_panel, gap_ok, layout_ok, index, sample_stroke, SHAPES,
                    POS_RANGE)
from .render import object_ink

C_RULES = ("C2b", "C3b", "C4", "C4b")
COUNT_TARGET = {"C2b": ("circle", 4), "C3b": ("square", 5)}
COUNT_REPLACE = {"C2b": ("square", "triangle", "pentagon", "cross"),
                 "C3b": ("circle", "triangle", "pentagon", "cross")}
MATCH_TOL = 0.04
RULE_CHECK_PX, RULE_MIN_PX = 224, 2.0
EXACT_TOL = 0.08
EXACT_FACTORS = (0.98, 1.0, 1.02)
EXACT_WEIGHTS = (1.0, 1.0, 1.0, 3.0, 1.0, 4.0, 4.0)
MATCH_OFFSETS = ((0.0, 0.0), (0.003, -0.002))
SIZE_GRID = (0.06, 0.08, 0.10, 0.12, 0.14, 0.16)
STROKE_GRID = (3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5)
PERIOD = {"circle": 1, "square": 90, "cross": 90, "triangle": 120, "pentagon": 72}
C4_FILLED = 2
C4_WEIGHTS = (3.0, 7.5, 8.0, 4.5, 0.0)
C4_FIT_SIZES = (0.03, 0.08, 0.13, 0.18, 0.23)
C4B_REPLACE = ("pentagon",)
C4B_FILLED = 3
C4B_WEIGHTS = (3.0, 7.5, 8.0, 4.5, 3.0)
FILLED_WEIGHTS = np.array([4.0, 4.0, 2.0, 2.0])
C4B_MIN_REPLACEMENT = 0.065

C_RULE_TYPE = {r: "C" for r in C_RULES}
C_RULE_PROGRAM = {
    "C2b": "count(o: o.shape == circle) == 4  (total count fixed within each twin pair)",
    "C3b": "count(o: o.shape == square) == 5  (total count fixed within each twin pair)",
    "C4": "count(o: o.fill == filled) == 2  (twin fills one more or un-fills one; sizes rebalance ink and outline)",
    "C4b": ("count(o: o.shape == circle and o.fill == filled) == 2  "
            "(3 filled circles or pentagons; twin keeps fill count, ink)"),
}
C_KNOB_NAME = {"C2b": "circle count pair (4 vs 3) at a fixed total of 5-6 shapes",
               "C3b": "square count pair (5 vs 4) at a fixed total of 6 shapes",
               "C4": "number of shapes in the panel (3-5)",
               "C4b": "number of shapes in the panel (4-6; 3 of them filled)"}
C_ID_BINS = {"C2b": [(4, 4)], "C3b": [(5, 5)], "C4": [(3, 3), (4, 4), (5, 5)], "C4b": [(4, 4), (5, 5), (6, 6)]}
C_FULL = {"C2b": (4, 4), "C3b": (5, 5), "C4": (3, 5), "C4b": (4, 6)}
C_TUNING = {"C2b": {"n_objects_id": (5, 6)},
            "C3b": {"n_objects_id": (6, 6), "size_range_id": (0.06, 0.12), "interior_target": True,
                    "exact_match": True},
            "C4": {"n_objects_id": None},
            "C4b": {"n_objects_id": None, "exact_match": True}}


def _u(rng, r):
    return float(rng.uniform(r[0], r[1]))


def _clone(objs):
    return [dict(o) for o in objs]


def _place(rng, shapes, fills, ranges, stroke, tries=60):
    # sequential placement with the minimum gap; shapes[k] / ranges[k] are the shape list and size range of object k
    objs = []
    for sh, f, sr in zip(shapes, fills, ranges):
        for _ in range(tries):
            o = make_obj(sh[int(rng.integers(len(sh)))], _u(rng, sr), f, _u(rng, POS_RANGE), _u(rng, POS_RANGE),
                         rng.uniform(0, 360), stroke, "distractor", mirror=bool(rng.random() < .5))
            if in_panel(o) and all(gap_ok(o, q) for q in objs):
                objs.append(o)
                break
        else:
            return None
    return objs


def count_shape(scene, shape):
    return sum(o["shape"] == shape for o in scene["objects"])


def c2b(scene, k=4):
    return count_shape(scene, "circle") == k


def c3b(scene, k=5):
    return count_shape(scene, "square") == k


def c4(scene, k=C4_FILLED):
    return sum(o["fill"] for o in scene["objects"]) == k


def filled_circles(scene):
    return sum(o["fill"] and o["shape"] == "circle" for o in scene["objects"])


def c4b(scene, k=2):
    return filled_circles(scene) == k


C_RULE_FN = {"C2b": c2b, "C3b": c3b, "C4": c4, "C4b": c4b}


def _crop_stats(o, style="default", crop=0.5):
    # one object at 128 px as in audit Track 1: grad weight, horizontal, diagonal, edge px, outline px, ink, ink^2
    px = int(crop * 512) * 4
    cv = Image.new("L", (px, px), 0)
    dr = ImageDraw.Draw(cv)
    g = object_ink(dict(o, x=o["x"] - 0.5 + crop / 2, y=o["y"] - 0.5 + crop / 2), style)
    for p in (list(g.geoms) if hasattr(g, "geoms") else [g]):
        dr.polygon([(x * 2048, y * 2048) for x, y in p.exterior.coords], fill=255)
        for r in p.interiors:
            dr.polygon([(x * 2048, y * 2048) for x, y in r.coords], fill=0)
    im = 255 - np.asarray(cv.reduce(4), np.uint8)
    q = np.asarray(Image.fromarray(im).reduce(4), float) / 255.0
    fg = q < 0.5
    gy, gx = np.gradient(q)
    mag = np.hypot(gx, gy)
    e = mag > 0.1
    a = np.mod(np.arctan2(gy, gx), np.pi)[e]
    w = mag[e]
    ink = 1.0 - q
    return np.array([w.sum(), (w * np.sin(a) ** 2).sum(), (w * np.abs(np.sin(2 * a))).sum(), e.sum(),
                     (fg & ~ndimage.binary_erosion(fg)).sum(), ink.sum(), (ink ** 2).sum()], float)


def _cell(size, stroke):
    return (int(np.argmin(np.abs(np.array(SIZE_GRID) - size))), int(np.argmin(np.abs(np.array(STROKE_GRID) - stroke))))


@lru_cache(maxsize=None)
def edge_table(shape, fill=False, cell=(2, 2), style="default"):
    # 128 px statistics per integer rotation over the symmetry period at one (size, stroke) grid cell, 2 offsets
    s, w = SIZE_GRID[cell[0]], STROKE_GRID[cell[1]]
    return np.array([np.mean([_crop_stats(make_obj(shape, s, fill, 0.5 + dx, 0.5 + dy, float(r), w), style)
                              for dx, dy in MATCH_OFFSETS], 0) for r in range(PERIOD[shape])])


def _direction_ratios(t):
    return np.c_[t[:, 1] / t[:, 0], t[:, 2] / t[:, 0]]


def matched_replacements(target, rotation, shapes, tol=MATCH_TOL, fill=False, cell=(2, 2), style="default"):
    # (shape, rotation) candidates matching the target's 128 px statistics (outline) or direction ratios (filled)
    f = _direction_ratios if fill else (lambda t: t)
    vt = f(edge_table(target, fill, cell, style))[int(round(rotation)) % PERIOD[target]]
    out = {}
    for sh in shapes:
        d = np.abs(f(edge_table(sh, fill, cell, style)) / vt - 1).max(1)
        ok = np.nonzero(d <= tol)[0]
        if len(ok):
            out[sh] = ok
    return out


def object_stats(o, style="default"):
    # 128 px statistics of one object rendered at its own pixel phase (shift quantised to 1/128 of the panel)
    dx, dy = np.floor((o["x"] - 0.5) * 128) / 128, np.floor((o["y"] - 0.5) * 128) / 128
    return _crop_stats(finalize(dict(o, x=o["x"] - dx, y=o["y"] - dy)), style)


def exact_replacement(o, cand, style, factors=EXACT_FACTORS):
    # the (shape, rotation, size factor) whose rendered statistics are closest to the removed object (weighted squares)
    v = object_stats(o, style)
    best = None
    for sh in sorted(cand):
        for r in cand[sh]:
            for k in factors:
                q = finalize(dict(o, shape=sh, rotation=0.0 if sh == "circle" else float(r), size=o["size"] * k,
                                  mirror=False))
                e = float(((np.array(EXACT_WEIGHTS) * (object_stats(q, style) / v - 1)) ** 2).sum())
                if best is None or e < best[0]:
                    best = (e, q)
    return best


def _filled_key(u):
    return np.array([u[1] / u[0], u[2] / u[0], u[5], u[6]])


def exact_filled_replacement(o, cand, size, style, factors=EXACT_FACTORS):
    # filled replacement (shape, rotation, size factor) closest to the circle in 128 px direction ratios and ink sums
    tv = _filled_key(object_stats(o, style))
    best = None
    for sh in sorted(cand):
        for r in cand[sh]:
            for k in factors:
                q = finalize(dict(o, shape=sh, rotation=float(r), size=size * k, mirror=False))
                e = float(((FILLED_WEIGHTS * (_filled_key(object_stats(q, style)) / tv - 1)) ** 2).sum())
                if best is None or e < best[0]:
                    best = (e, q)
    return best


def _count_by_shape(rng, st):
    # exactly k target shapes among n outline shapes; twin turns one target into an edge-matched other shape in place
    target, k = COUNT_TARGET[st["rule"]]
    stroke = sample_stroke(rng, st["style"])
    n = int(rng.integers(st["n_objects"][0], st["n_objects"][1] + 1))
    others = [s for s in SHAPES if s != target]
    objs = _place(rng, [[target]] * k + [others] * (n - k), [False] * n, [st["size_range"]] * n, stroke)
    if objs is None:
        return None
    objs = [objs[i] for i in rng.permutation(n)]
    exact = st.get("exact_match", False)
    tol = st.get("match_tol", EXACT_TOL if exact else MATCH_TOL)
    tg = [i for i, o in enumerate(objs) if o["shape"] == target
          and not (st.get("interior_target") and not _inside_others(objs, i, o))]
    cand = {i: matched_replacements(target, objs[i]["rotation"], COUNT_REPLACE[st["rule"]], tol,
                                    cell=_cell(objs[i]["size"], stroke), style=st["style"]) for i in tg}
    tg = [i for i in tg if cand[i]]
    if not tg:
        return None
    j = tg[int(rng.integers(len(tg)))]
    objs[j] = dict(objs[j], role="t1")
    pos = index(_clone(objs))
    twin = _clone(pos)
    if exact:
        twin[j] = exact_replacement(pos[j], cand[j], st["style"])[1]
    else:
        shs = sorted(cand[j])
        new = shs[int(rng.integers(len(shs)))]
        rots = cand[j][new]
        rot = 0.0 if new == "circle" else float((rots[int(rng.integers(len(rots)))] + rng.uniform(-0.5, 0.5)) % 360)
        twin[j] = finalize(dict(twin[j], shape=new, mirror=False, rotation=rot))
    if not layout_ok(twin) or (st.get("interior_target") and not _inside_others(twin, j, pos[j])):
        return None
    if poly(pos[j]).hausdorff_distance(poly(twin[j])) * RULE_CHECK_PX < RULE_MIN_PX:
        return None
    dec = {(j, f) for f in ("shape", "rotation", "size") if twin[j][f] != pos[j][f]}
    return pos, twin, k, dec, {"replacement": twin[j]["shape"]}


def _inside_others(objs, j, old):
    # both the removed and the new object lie inside the convex hull of the other objects (hull and bbox unchanged)
    from shapely.ops import unary_union
    h = unary_union([poly(o) for i, o in enumerate(objs) if i != j]).convex_hull
    return h.contains(poly(objs[j])) and h.contains(poly(old))


def _ink_style(o, style):
    return object_ink(o, style)


def _quad_fit(o, style):
    # per-object cues as polynomials in size: style ink area, style ink boundary length, default ink area
    xs = np.array(C4_FIT_SIZES)
    rows = []
    for s in xs:
        q = finalize(dict(o, size=float(s)))
        g = _ink_style(q, style)
        rows.append([g.area, g.boundary.length, ink_geom(q).area])
    Y = np.array(rows)
    return [np.polyfit(xs, Y[:, c], 2) for c in range(3)]


def _hull_area(objs, sizes, unit):
    P = np.vstack([np.array([o["x"], o["y"]]) + s * u for o, s, u in zip(objs, sizes, unit)])
    return float(ConvexHull(P).volume)


def c4_cues(objs, style="default"):
    # exact C4 balance cues: style ink, style ink boundary, size sum, field hull, ink concentration, default ink
    from shapely.ops import unary_union
    g = [_ink_style(o, style) for o in objs]
    m = np.array([x.area for x in g])
    return np.array([m.sum(), sum(x.boundary.length for x in g), sum(o["size"] for o in objs),
                     unary_union([poly(o) for o in objs]).convex_hull.area, (m ** 2).sum() / m.sum() ** 2,
                     sum(ink_geom(o).area for o in objs)])


def c4_balance(pos, twin, bounds, style, weights=C4_WEIGHTS):
    # twin sizes minimising the weighted relative mismatch of the C4 cues to the positive; returns sizes, residual
    target = c4_cues(pos, style)
    fits = [_quad_fit(o, style) for o in twin]
    unit = []
    for o in twin:
        P = np.array(poly(finalize(dict(o, size=1.0))).exterior.coords)
        unit.append(P - np.array([o["x"], o["y"]]))
    w = np.array(weights)
    cols = list(range(len(weights)))

    def res(s):
        m = np.array([np.polyval(f[0], x) for f, x in zip(fits, s)])
        B = sum(np.polyval(f[1], x) for f, x in zip(fits, s))
        D = sum(np.polyval(f[2], x) for f, x in zip(fits, s))
        v = np.array([m.sum(), B, s.sum(), _hull_area(twin, s, unit), (m ** 2).sum() / m.sum() ** 2, D])
        return w * (v[cols] - target[cols]) / target[cols]

    lo, hi = np.array(bounds, float).T
    s0 = np.clip([o["size"] for o in twin], lo, hi)
    r = least_squares(res, s0, bounds=(lo, hi), xtol=1e-9, ftol=1e-9)
    return [float(x) for x in r.x], float(np.sqrt(np.mean(r.fun ** 2)))


def _c4(rng, st):
    # exactly 2 filled among n; twin fills one more or un-fills one (50/50), then all sizes rebalance the cues
    stroke = sample_stroke(rng, st["style"])
    lo, hi = st["n_objects"] if st.get("n_objects") else st["knob"]
    n = int(rng.integers(lo, hi + 1))
    fr, orng = st.get("c4_fill_range", st["size_range"]), st.get("c4_outline_range", st["size_range"])
    fills = [True] * C4_FILLED + [False] * (n - C4_FILLED)
    objs = _place(rng, [SHAPES] * n, fills, [fr if f else orng for f in fills], stroke)
    if objs is None:
        return None
    objs = [objs[i] for i in rng.permutation(n)]
    up = bool(rng.integers(2))
    cand = [i for i, o in enumerate(objs) if o["fill"] != up]
    w = st.get("c4_weights", C4_WEIGHTS)
    best = None
    for j in ([cand[int(rng.integers(len(cand)))]] if not st.get("c4_best_j") else cand):
        tw = _clone(objs)
        tw[j] = dict(tw[j], fill=up)
        bnd = [fr if o["fill"] else orng for o in tw]
        sizes, err = c4_balance(objs, tw, bnd, st["style"], w)
        if best is None or err < best[2]:
            best = (j, sizes, err)
    j, sizes, err = best
    if err > st.get("c4_max_residual", np.inf):
        return None
    objs[j] = dict(objs[j], role="t1")
    pos = index(_clone(objs))
    twin = _clone(pos)
    twin[j] = dict(twin[j], fill=up)
    twin = [finalize(dict(o, size=s)) for o, s in zip(twin, sizes)]
    if not layout_ok(twin):
        return None
    dec = {(j, "fill")} | {(i, "size") for i, o in enumerate(twin) if abs(o["size"] - pos[i]["size"]) >= 1e-9}
    return pos, twin, n, dec, {"direction": "up" if up else "down", "residual": err}


def _c4b(rng, st):
    # 2 filled circles + 1 filled pentagon + outlines; twin: a filled circle becomes an edge-matched filled pentagon
    from .scene import size_for_ink, ink_area
    stroke = sample_stroke(rng, st["style"])
    lo, hi = st["n_objects"] if st.get("n_objects") else st["knob"]
    n = int(rng.integers(lo, hi + 1))
    shapes = [["circle"]] * 2 + [list(C4B_REPLACE)] + [SHAPES] * (n - C4B_FILLED)
    fills = [True] * C4B_FILLED + [False] * (n - C4B_FILLED)
    objs = _place(rng, shapes, fills, [st["size_range"]] * n, stroke)
    if objs is None:
        return None
    objs = [objs[i] for i in rng.permutation(n)]
    tg = [i for i, o in enumerate(objs) if o["fill"] and o["shape"] == "circle"]
    j = tg[int(rng.integers(len(tg)))]
    objs[j] = dict(objs[j], role="t1")
    pos = index(_clone(objs))
    exact = st.get("exact_match", False)
    cand = matched_replacements("circle", 0.0, C4B_REPLACE, st.get("match_tol", EXACT_TOL if exact else MATCH_TOL),
                                fill=True, cell=_cell(pos[j]["size"], stroke), style=st["style"])
    twin = _clone(pos)
    if exact:
        sz = {sh: size_for_ink(sh, ink_area(pos[j]), True, stroke) for sh in cand}
        twin[j] = min((exact_filled_replacement(pos[j], {sh: cand[sh]}, sz[sh], st["style"]) for sh in sorted(cand)),
                      key=lambda b: b[0])[1]
    else:
        shs = sorted(cand)
        new = shs[int(rng.integers(len(shs)))]
        rots = cand[new]
        rot = float((rots[int(rng.integers(len(rots)))] + rng.uniform(-0.5, 0.5)) % 360)
        size = size_for_ink(new, ink_area(pos[j]), True, stroke)
        twin[j] = finalize(dict(twin[j], shape=new, size=size, rotation=rot, mirror=False))
    new = twin[j]["shape"]
    w = tuple(st.get("c4_weights", C4B_WEIGHTS))
    w = w + (w[0],) if st["style"] != "default" else w
    bnd = [(max(st["size_range"][0], C4B_MIN_REPLACEMENT), st["size_range"][1]) if i == j else st["size_range"]
           for i in range(n)]
    sizes, err = c4_balance(pos, twin, bnd, st["style"], w)
    twin = [finalize(dict(o, size=x)) for o, x in zip(twin, sizes)]
    if not layout_ok(twin) or (st.get("interior_target") and not _inside_others(twin, j, pos[j])):
        return None
    dec = {(j, "shape"), (j, "rotation")} | {(i, "size") for i, o in enumerate(twin)
                                              if abs(o["size"] - pos[i]["size"]) >= 1e-9}
    return pos, twin, n, dec, {"replacement": new, "residual": err}


C_SAMPLERS = {"C2b": _count_by_shape, "C3b": _count_by_shape, "C4": _c4, "C4b": _c4b}
C_INK_CHECKED = ("C2b", "C3b", "C4b")


def c_candidates():
    # 5c candidates for the C rules (C4 fill counts are already in the pilot list)
    C = {}
    for k in range(0, 5):
        C[f"C4b:filled circle count=={k}"] = lambda s, k=k: filled_circles(s) == k
        C[f"C4b:filled circle count>={k}"] = lambda s, k=k: filled_circles(s) >= k
    for r, (sh, _) in COUNT_TARGET.items():
        for k in range(0, 9):
            C[f"{r}:{sh} count=={k}"] = lambda s, sh=sh, k=k: count_shape(s, sh) == k
            C[f"{r}:{sh} count>={k}"] = lambda s, sh=sh, k=k: count_shape(s, sh) >= k
    return C


C_FAMILY = {"C2b": ("C2b:",), "C3b": ("C3b:",), "C4": ("C4:",), "C4b": ("C4b:",)}
C_CODE_BASE = 700
