import copy
import numpy as np
from shapely import affinity
from . import scene as S
from .scene import (make_obj, moved, finalize, poly, ink_area, ink_geom, layout_ok, in_panel, gap_ok,
                    add_distractors, index, sample_stroke, size_for_ink, SHAPES, MIN_GAP, EDGE_MARGIN, POS_RANGE)
from .predicates import RULE_FN, truth, nest_pairs, touch_pairs, ink_gap

MAX_DRAWS = 20000
FIELDS = ("shape", "fill", "x", "y", "size", "rotation", "mirror", "stroke_px", "role")
INK_TOL = 0.01


class SamplerError(RuntimeError):
    pass


def _n(rng, st):
    lo, hi = st["n_objects"]
    return int(rng.integers(lo, hi + 1))


def _u(rng, r):
    return float(rng.uniform(r[0], r[1]))


def _clone(objs):
    return [dict(o) for o in objs]


def _a1(rng, st):
    stroke = sample_stroke(rng, st["style"])
    lo, hi = max(st["knob"][0], st["size_range"][0]), min(st["knob"][1], st["size_range"][1])
    t = S.random_obj(rng, SHAPES, (lo, hi), stroke, fill=True, role="t1")
    if not in_panel(t):
        return None
    n = _n(rng, st)
    objs = add_distractors(rng, [t], n - 1, SHAPES, st["size_range"], stroke, fills=[False] * (n - 1))
    if objs is None:
        return None
    pos = index(_clone(objs))
    twin = _clone(pos)
    twin[0] = dict(twin[0], fill=False)
    return pos, twin, t["size"], {(0, "fill")}, {}


def a6_square_size(tri_size, stroke):
    # square size so that (filled tri + outline sq) and (outline tri + filled sq) carry equal ink
    T = make_obj("triangle", tri_size, True, .5, .5, 0, stroke)
    d = ink_area(T) - ink_area(dict(T, fill=False))
    lo, hi = 0.01, 0.40
    for _ in range(40):
        m = 0.5 * (lo + hi)
        q = make_obj("square", m, True, .5, .5, 0, stroke)
        e = ink_area(q) - ink_area(dict(q, fill=False))
        lo, hi = (m, hi) if e < d else (lo, m)
    return 0.5 * (lo + hi)


A6_DISTRACTORS = {"FS": ("square", True), "OT": ("triangle", False), "OS": ("square", False)}


def _a6(rng, st):
    stroke = sample_stroke(rng, st["style"])
    sr = st["size_range"]
    st_ = _u(rng, sr)
    ss = a6_square_size(st_, stroke)
    if not sr[0] <= ss <= sr[1]:
        return None
    T = make_obj("triangle", st_, True, _u(rng, POS_RANGE), _u(rng, POS_RANGE), rng.uniform(0, 360), stroke, "t1")
    Q = make_obj("square", ss, False, _u(rng, POS_RANGE), _u(rng, POS_RANGE), rng.uniform(0, 360), stroke, "t2")
    if not layout_ok([T, Q]):
        return None
    k = int(rng.integers(st["knob"][0], st["knob"][1] + 1))
    pool = st.get("a6_pool", ["FS", "OT"])
    kinds = [pool[int(rng.integers(len(pool)))] for _ in range(k)]
    objs = add_distractors(rng, [T, Q], k, [[A6_DISTRACTORS[c][0]] for c in kinds], sr, stroke,
                           fills=[A6_DISTRACTORS[c][1] for c in kinds])
    if objs is None:
        return None
    pos = index(_clone(objs))
    twin = _clone(pos)
    twin[0] = dict(twin[0], fill=False)
    twin[1] = dict(twin[1], fill=True)
    return pos, twin, k, {(0, "fill"), (1, "fill")}, {}


B1_DISTRACTOR_SHAPES = [s for s in SHAPES if s not in ("triangle", "circle")]


def _extent(o):
    b = poly(o).bounds
    return o["y"] - b[1], b[3] - o["y"]


def equal_ink_partner(rng, shape1, shape2, size_range, fill, stroke, jitter=0.0, offset=0.0):
    s1 = _u(rng, size_range)
    a = ink_area(make_obj(shape1, s1, fill, .5, .5, 0, stroke)) * (1 + offset + rng.uniform(-jitter, jitter))
    s2 = size_for_ink(shape2, a, fill, stroke)
    if not size_range[0] <= s2 <= size_range[1]:
        return None
    return s1, s2


_GRID = np.arange(0, 360, 0.05)


def _extent_table(shape):
    v = S.UNIT[shape]
    t = np.deg2rad(_GRID)[:, None]
    w = v[None, :, 0] * np.sin(t) + v[None, :, 1] * np.cos(t)
    return w.max(1) - w.min(1)


_EXT = {}


def rotation_for_extent(rng, shape, size, target_h):
    # a rotation at which the shape's vertical extent equals target_h (random among the solutions)
    if shape not in _EXT:
        _EXT[shape] = _extent_table(shape)
    d = _EXT[shape] * size - target_h
    idx = np.nonzero(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    if len(idx) == 0:
        return None
    i = idx[int(rng.integers(len(idx)))]
    return float(_GRID[i] - d[i] * (_GRID[i + 1] - _GRID[i]) / (d[i + 1] - d[i]))


def _b1(rng, st):
    # twin swaps the two roles' ink-centroid y (x kept); extent predicate checked on both panels
    stroke = sample_stroke(rng, st["style"])
    fill = bool(rng.random() < 0.5) if st.get("role_fill") != "outline" else False
    sz = equal_ink_partner(rng, "circle", "triangle", st["size_range"], fill, stroke,
                           st.get("role_ink_jitter", 0.0), st.get("role_ink_offset", 0.0))
    if sz is None:
        return None
    sc, stri = sz
    T = make_obj("triangle", stri, fill, _u(rng, POS_RANGE), .5, rng.uniform(0, 360), stroke, "t1")
    C = make_obj("circle", sc, fill, _u(rng, POS_RANGE), .5, 0, stroke, "t2")
    (tu, td), (cu, cd) = _extent(T), _extent(C)
    g = _u(rng, st["knob"])
    H = tu + td + g + cu + cd
    if H > 1 - 2 * EDGE_MARGIN:
        return None
    top = rng.uniform(EDGE_MARGIN, 1 - EDGE_MARGIN - H)
    yT, yC = top + tu, top + tu + td + g + cu
    pos = [moved(T, y=yT), moved(C, y=yC)]
    twn = [moved(T, y=yC), moved(C, y=yT)]
    if not (layout_ok(pos) and layout_ok(twn)):
        return None
    if not twn[0]["bbox"][1] > twn[1]["bbox"][3]:
        return None
    n = _n(rng, st)
    objs = add_distractors(rng, pos, n - 2, B1_DISTRACTOR_SHAPES, st["size_range"], stroke)
    if objs is None:
        return None
    pos = index(_clone(objs))
    twin = index(twn + _clone(objs[2:]))
    if not layout_ok(twin):
        return None
    g_true = pos[1]["bbox"][1] - pos[0]["bbox"][3]
    return pos, twin, g_true, {(0, "y"), (1, "y")}, {}


def _inscribed():
    from shapely.ops import polylabel
    from shapely.geometry import Polygon
    out = {}
    for k, v in S.UNIT.items():
        P = Polygon(v)
        c = polylabel(P, tolerance=1e-4)
        out[k] = (float(P.exterior.distance(c)), (float(c.x), float(c.y)))
    return out


INSCRIBED = _inscribed()


def _pole(o):
    r, (u, v) = INSCRIBED[o["shape"]]
    if o.get("mirror"):
        u = -u
    t = np.deg2rad(o["rotation"])
    return (o["x"] + o["size"] * (u * np.cos(t) - v * np.sin(t)),
            o["y"] - o["size"] * (u * np.sin(t) + v * np.cos(t)), r * o["size"])


def _place_outside(rng, objs, I, st, tries=300):
    near = st.get("b3_twin_near")
    O = poly(objs[0])
    box = [min(o["bbox"][0] for o in objs), min(o["bbox"][1] for o in objs),
           max(o["bbox"][2] for o in objs), max(o["bbox"][3] for o in objs)]
    for _ in range(tries):
        c = moved(I, x=_u(rng, POS_RANGE), y=_u(rng, POS_RANGE))
        if not in_panel(c):
            continue
        if st.get("twin_within_bbox"):
            b = c["bbox"]
            if b[0] < box[0] or b[1] < box[1] or b[2] > box[2] or b[3] > box[3]:
                continue
        q = poly(c)
        if any(poly(o).distance(q) < MIN_GAP for o in objs):
            continue
        if near is not None and O.distance(q) > near:
            continue
        return c
    return None


def _b3(rng, st):
    # constructive: inner circumcircle placed inside the outer's largest inscribed disk shrunk by the margin
    stroke = sample_stroke(rng, st["style"])
    sr = st["size_range"]
    outer_range = st.get("b3_outer_range", sr)
    m = _u(rng, st["knob"])
    ish = SHAPES[int(rng.integers(len(SHAPES)))]
    I = make_obj(ish, rng.uniform(sr[0], sr[0] + 0.03), False, .5, .5, rng.uniform(0, 360), stroke, "t2",
                 mirror=bool(rng.random() < .5))
    need = I["radius"] + m
    feas = [k for k in SHAPES if need / INSCRIBED[k][0] <= outer_range[1]]
    if not feas:
        return None
    osh = feas[int(rng.integers(len(feas)))]
    so = rng.uniform(max(need / INSCRIBED[osh][0], outer_range[0]), outer_range[1])
    O = make_obj(osh, so, False, _u(rng, POS_RANGE), _u(rng, POS_RANGE), rng.uniform(0, 360), stroke, "t1",
                 mirror=bool(rng.random() < .5))
    if not in_panel(O):
        return None
    px, py, rin = _pole(O)
    slack = rin - need
    a, rr = rng.uniform(0, 2 * np.pi), slack * np.sqrt(rng.random())
    I = moved(I, x=px + rr * np.cos(a), y=py + rr * np.sin(a))
    P, q = poly(O), poly(I)
    if not P.contains(q):
        return None
    m_true = P.exterior.distance(q)
    if not st["knob"][0] <= m_true <= st["knob"][1]:
        return None
    tri = _u(rng, sr)
    sq = a6_square_size(tri, stroke)
    if not sr[0] <= sq <= sr[1]:
        return None
    objs = [O]
    for d in (make_obj("triangle", tri, bool(rng.random() < .5), _u(rng, POS_RANGE), _u(rng, POS_RANGE),
                       rng.uniform(0, 360), stroke),
              make_obj("square", sq, bool(rng.random() < .5), _u(rng, POS_RANGE), _u(rng, POS_RANGE),
                       rng.uniform(0, 360), stroke)):
        ok = False
        for _ in range(60):
            if in_panel(d) and all(gap_ok(d, o) for o in objs):
                ok = True
                break
            d = moved(d, x=_u(rng, POS_RANGE), y=_u(rng, POS_RANGE))
        if not ok:
            return None
        objs.append(d)
    objs = add_distractors(rng, objs, max(_n(rng, st) - 4, 0), SHAPES, sr, stroke)
    if objs is None:
        return None
    outside = _place_outside(rng, objs, I, st)
    if outside is None:
        return None
    pos = index(_clone([objs[0], I] + objs[1:]))
    twin = index(_clone([objs[0], outside] + objs[1:]))
    if not (layout_ok(pos, exempt=[(0, 1)]) and layout_ok(twin)):
        return None
    return pos, twin, m_true, {(1, "x"), (1, "y")}, {"slot_T": 2, "slot_S": 3}


def _contact_t(P1, I, c1, u, t_hi):
    lo, hi = 0.0, t_hi
    for _ in range(50):
        m = 0.5 * (lo + hi)
        q = poly(moved(I, x=c1[0] + m * u[0], y=c1[1] + m * u[1]))
        lo, hi = (m, hi) if P1.intersects(q) else (lo, m)
    return hi


def _gap_t(P1, I, c1, u, t0, target):
    lo, hi = t0, t0 + target + 0.5
    for _ in range(50):
        m = 0.5 * (lo + hi)
        q = poly(moved(I, x=c1[0] + m * u[0], y=c1[1] + m * u[1]))
        lo, hi = (m, hi) if P1.distance(q) < target else (lo, m)
    return hi


def _spanning_distractors(rng, objs, roles, size_range, stroke, tries=80):
    # two distractors beyond opposite corners of the role box (both panels), so they set the global extents
    rb = (min(o["bbox"][0] for o in roles), min(o["bbox"][1] for o in roles),
          max(o["bbox"][2] for o in roles), max(o["bbox"][3] for o in roles))
    diag = bool(rng.random() < 0.5)
    corners = [(True, True), (False, False)] if diag else [(False, True), (True, False)]
    out = list(objs)
    if rb[0] <= POS_RANGE[0] or rb[1] <= POS_RANGE[0] or rb[2] >= POS_RANGE[1] or rb[3] >= POS_RANGE[1]:
        return None
    for left, top in corners:
        for _ in range(tries):
            d = S.random_obj(rng, SHAPES, size_range, stroke)
            x = rng.uniform(POS_RANGE[0], rb[0]) if left else rng.uniform(rb[2], POS_RANGE[1])
            y = rng.uniform(POS_RANGE[0], rb[1]) if top else rng.uniform(rb[3], POS_RANGE[1])
            d = moved(d, x=x, y=y)
            b = d["bbox"]
            okx = b[0] <= rb[0] if left else b[2] >= rb[2]
            oky = b[1] <= rb[1] if top else b[3] >= rb[3]
            if okx and oky and in_panel(d) and all(gap_ok(d, o) for o in out + roles[2:]):
                out.append(d)
                break
        else:
            return None
    return out


def _b4(rng, st):
    stroke = sample_stroke(rng, st["style"])
    fill = bool(rng.random() < 0.5)
    sh1, sh2 = SHAPES[int(rng.integers(len(SHAPES)))], SHAPES[int(rng.integers(len(SHAPES)))]
    sz = equal_ink_partner(rng, sh1, sh2, st["size_range"], fill, stroke)
    if sz is None:
        return None
    A = make_obj(sh1, sz[0], fill, _u(rng, POS_RANGE), _u(rng, POS_RANGE), rng.uniform(0, 360), stroke, "t1",
                 mirror=bool(rng.random() < .5))
    B = make_obj(sh2, sz[1], fill, .5, .5, rng.uniform(0, 360), stroke, "t2", mirror=bool(rng.random() < .5))
    phi = rng.uniform(0, 2 * np.pi)
    u = (np.cos(phi), np.sin(phi))
    c1 = (A["x"], A["y"])
    P1 = poly(A)
    t_hi = A["radius"] + B["radius"] + 0.01
    tc = _contact_t(P1, B, c1, u, t_hi)
    w = S.stroke_w(A)
    if st.get("stroke_contact_eps_px") is not None:
        tc = _gap_t(P1, B, c1, u, tc, w - st["stroke_contact_eps_px"] / S.PANEL_PX)
    Bp = moved(B, x=c1[0] + tc * u[0], y=c1[1] + tc * u[1])
    lim = st.get("max_contact_overlap")
    if lim is not None and ink_geom(A).intersection(ink_geom(Bp)).area > lim * S.stroke_w(A) ** 2:
        return None
    g = _u(rng, st["knob"])
    tg = _gap_t(P1, B, c1, u, tc, g + w)
    Bt = moved(B, x=c1[0] + tg * u[0], y=c1[1] + tg * u[1])
    At = A
    if st.get("symmetric_separation"):
        h = 0.5 * (tg - tc)
        At = moved(A, x=A["x"] - h * u[0], y=A["y"] - h * u[1])
        Bt = moved(B, x=c1[0] + (tc + h) * u[0], y=c1[1] + (tc + h) * u[1])
    if not (in_panel(A) and in_panel(Bp) and in_panel(Bt) and in_panel(At)):
        return None
    n = _n(rng, st)
    if st.get("distractors_span_roles") and n > 2:
        objs = _spanning_distractors(rng, [A, Bp], [A, Bp, At, Bt], st["size_range"], stroke)
        if objs is None:
            return None
        objs = add_distractors(rng, objs, n - 4, SHAPES, st["size_range"], stroke)
    else:
        objs = add_distractors(rng, [A, Bp], n - 2, SHAPES, st["size_range"], stroke)
    if objs is None:
        return None
    if any(poly(d).distance(poly(Bt)) < MIN_GAP or poly(d).distance(poly(At)) < MIN_GAP for d in objs[2:]):
        return None
    pos = index(_clone(objs))
    twin = index(_clone([At, Bt] + objs[2:]))
    if not (layout_ok(pos, exempt=[(0, 1)]) and layout_ok(twin, exempt=[(0, 1)])):
        return None
    gap_true = ink_gap({"objects": twin}, 0, 1)
    dec = {(1, "x"), (1, "y")} | ({(0, "x"), (0, "y")} if st.get("symmetric_separation") else set())
    return pos, twin, gap_true, dec, {}


def _scale_all(objs, k):
    return [finalize(dict(o, size=o["size"] * k)) for o in objs]


def _hull(objs):
    from shapely.ops import unary_union
    return unary_union([poly(o) for o in objs]).convex_hull.area


def _c1(rng, st, control):
    # 3 vs 4 shapes, midpoint design: sizes and layout scale symmetrically about a midpoint, sign c per pair
    if control not in ("up", "down"):
        return _c1_simple(rng, st, control)
    stroke = sample_stroke(rng, st["style"])
    sr = st["size_range"]
    c = 1.0 if control == "up" else -1.0
    ls = c * rng.uniform(*st["c1_size_shift"]) - st["c1_size_offset"]
    lh = c * rng.uniform(*st["c1_layout_shift"]) + st["c1_layout_offset"]
    m = float(np.exp(rng.uniform(np.log(sr[0]), np.log(sr[1]))))
    sp, sn = m * np.exp(-ls / 2), m * np.exp(ls / 2)
    if not (sr[0] <= sp <= sr[1] and sr[0] <= sn <= sr[1]):
        return None
    pts = rng.uniform(POS_RANGE[0], POS_RANGE[1], (3, 2))
    box = (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
    p4 = np.array([rng.uniform(box[0], box[2]), rng.uniform(box[1], box[3])])
    allp = np.vstack([pts, p4])
    cen = allp.mean(0)
    rp, rn = np.exp(-lh / 4), np.exp(lh / 4)
    look = []
    for k in range(4):
        look.append((SHAPES[int(rng.integers(len(SHAPES)))], bool(rng.random() < .5), rng.uniform(0, 360),
                     bool(rng.random() < .5)))
    def build(scale, size, n):
        out = []
        for k in range(n):
            q = cen + scale * (allp[k] - cen)
            sh, f, rot, mir = look[k]
            out.append(make_obj(sh, size, f, q[0], q[1], rot, stroke, "distractor", mirror=mir))
        return out
    pos = build(rp, sp, 3)
    twin = build(rn, sn, 4)
    if not (layout_ok(pos) and layout_ok(twin)):
        return None
    declared = {(3, "*")} | {(i, f) for i in range(3) for f in ("size", "x", "y")}
    return index(pos), index(twin), 3, declared, {"control": control}


def _c1_simple(rng, st, control):
    # earlier C1 designs (equal_size / equal_ink / inverse_ink), kept for the logged audit runs
    stroke = sample_stroke(rng, st["style"])
    sr = st["size_range"]
    s = _u(rng, sr)
    objs = []
    for k in range(3):
        sh = SHAPES[int(rng.integers(len(SHAPES)))]
        objs.append(make_obj(sh, s, bool(rng.random() < .5), _u(rng, POS_RANGE), _u(rng, POS_RANGE),
                             rng.uniform(0, 360), stroke, "distractor", mirror=bool(rng.random() < .5)))
    if not layout_ok(objs):
        return None
    box = np.array([min(o["bbox"][0] for o in objs), min(o["bbox"][1] for o in objs),
                    max(o["bbox"][2] for o in objs), max(o["bbox"][3] for o in objs)])
    new = None
    for _ in range(60):
        sh = SHAPES[int(rng.integers(len(SHAPES)))]
        d = make_obj(sh, s, bool(rng.random() < .5), rng.uniform(box[0], box[2]), rng.uniform(box[1], box[3]),
                     rng.uniform(0, 360), stroke, mirror=bool(rng.random() < .5))
        bb = d["bbox"]
        if st.get("twin_within_bbox") and (bb[0] < box[0] or bb[1] < box[1] or bb[2] > box[2] or bb[3] > box[3]):
            continue
        if in_panel(d) and all(gap_ok(d, o) for o in objs):
            new = d
            break
    if new is None:
        return None
    pos = index(_clone(objs))
    twin = _clone(objs) + [new]
    declared = {(3, "*")} | {(i, f) for i in range(3) for f in ("size", "x", "y")}
    if control in ("up", "down"):
        sgn = 1.0 if control == "up" else -1.0
        ks = st["c1_size_" + control]
        kh = st["c1_layout_" + control]
        k = float(np.exp(sgn * rng.uniform(*ks)))
        rho = float(np.exp(sgn * rng.uniform(*kh)))
        cx = np.mean([o["x"] for o in twin])
        cy = np.mean([o["y"] for o in twin])
        twin = [finalize(dict(o, size=o["size"] * k, x=cx + rho * (o["x"] - cx), y=cy + rho * (o["y"] - cy)))
                for o in twin]
    elif control == "equal_ink":
        target = sum(ink_area(o) for o in pos)
        lo, hi = 0.5, 1.0
        for _ in range(40):
            m = 0.5 * (lo + hi)
            tot = sum(ink_area(o) for o in _scale_all(twin, m))
            lo, hi = (m, hi) if tot < target else (lo, m)
        twin = _scale_all(twin, 0.5 * (lo + hi))
        declared = {(3, "*")} | {(i, "size") for i in range(3)}
    else:
        declared = {(3, "*")}
    if not all(sr[0] <= o["size"] <= sr[1] for o in twin):
        return None
    twin = index(twin)
    if not layout_ok(twin):
        return None
    return pos, twin, 3, declared, {"control": control}


C1B_REPLACE = ("circle", "square", "pentagon", "cross")
NON_TRIANGLES = [x for x in SHAPES if x != "triangle"]


def _c1b(rng, st):
    # exactly 3 triangles among n outline shapes; twin turns one triangle into a non-chiral non-triangle in place
    stroke = sample_stroke(rng, st["style"])
    n = _n(rng, st)
    sr = st["size_range"]
    objs = []
    for k in range(n):
        sh = "triangle" if k < 3 else NON_TRIANGLES[int(rng.integers(len(NON_TRIANGLES)))]
        for _ in range(60):
            o = make_obj(sh, _u(rng, sr), False, _u(rng, POS_RANGE), _u(rng, POS_RANGE), rng.uniform(0, 360), stroke,
                         "distractor", mirror=bool(rng.random() < .5))
            if in_panel(o) and all(gap_ok(o, q) for q in objs):
                objs.append(o)
                break
        else:
            return None
    order = rng.permutation(n)
    objs = [objs[i] for i in order]
    tri = [i for i, o in enumerate(objs) if o["shape"] == "triangle"]
    j = tri[int(rng.integers(len(tri)))]
    objs[j] = dict(objs[j], role="t1")
    pos = index(_clone(objs))
    twin = _clone(pos)
    new = C1B_REPLACE[int(rng.integers(len(C1B_REPLACE)))]
    twin[j] = finalize(dict(twin[j], shape=new, mirror=False, rotation=0.0 if new == "circle" else twin[j]["rotation"]))
    if not layout_ok(twin):
        return None
    dec = {(j, "shape")} | ({(j, "rotation")} if new == "circle" and pos[j]["rotation"] != 0.0 else set())
    return pos, twin, 3, dec, {}


SAMPLERS = {"A1": _a1, "A6": _a6, "B1": _b1, "B3": _b3, "B4": _b4, "C1b": _c1b}


def check_twin(rule, pos, twin, declared, ink_tol=INK_TOL, control=None):
    # returns a list of violations (empty = accepted)
    v = []
    P, T = pos["objects"], twin["objects"]
    add = any(f == "*" for _, f in declared)
    if len(T) != len(P) + (1 if add else 0):
        v.append("object count changed")
    for i, o in enumerate(P):
        if i >= len(T):
            break
        for f in FIELDS:
            a, b = o[f], T[i][f]
            same = abs(a - b) < 1e-9 if isinstance(a, float) else a == b
            if not same and (i, f) not in declared:
                v.append(f"undeclared change obj{i}.{f}")
            if same and (i, f) in declared and f not in ("size", "x", "y") and not add:
                v.append(f"declared field unchanged obj{i}.{f}")
    if not RULE_FN[rule](pos):
        v.append("positive fails rule")
    if RULE_FN[rule](twin):
        v.append("twin still satisfies rule")
    check_ink = rule in ("A6", "B1", "B3", "B4", "C1b", "A2") or (rule == "C1" and control == "equal_ink")
    if rule == "C1" and control == "inverse_ink":
        check_ink = False
    if check_ink:
        ip = sum(ink_area(o) for o in P)
        it = sum(ink_area(o) for o in T)
        if abs(ip - it) / ip > ink_tol:
            v.append(f"ink changed {ip:.5f}->{it:.5f}")
    return v


def _finish(objs, style):
    sc = {"objects": objs, "render_style": style}
    sc["truth"] = truth(sc)
    sc.pop("_polys", None)
    return sc


def sample_pair(st, seed, control=None):
    # rejection sampling from one seed; deterministic; returns pair dict with draw count
    rng = np.random.default_rng(seed)
    rule = st["rule"]
    if st.get("n_objects"):
        n = int(rng.integers(st["n_objects"][0], st["n_objects"][1] + 1))
        st = dict(st, n_objects=(n, n))
    for draws in range(1, MAX_DRAWS + 1):
        r = _c1(rng, st, control) if rule == "C1" else SAMPLERS[rule](rng, st)
        if r is None:
            continue
        pos, twin, knob, declared, meta = r
        P, T = _finish(pos, st["style"]), _finish(twin, st["style"])
        if check_twin(rule, P, T, declared, control=control):
            continue
        return {"pos": P, "twin": T, "knob": float(knob), "draws": draws, "seed": int(seed),
                "setting_id": st["id"], "meta": meta}
    raise SamplerError(f"{st['id']}: no pair after {MAX_DRAWS} draws (seed {seed})")
