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


# ---------- A1: some shape is filled; twin un-fills it
def _a1(rng, st):
    stroke = sample_stroke(rng)
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


# ---------- A6: a filled triangle is present; twin swaps fills between triangle and square
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
    stroke = sample_stroke(rng)
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


# ---------- B1: triangle entirely above circle; twin swaps vertical order keeping gap and x
B1_DISTRACTOR_SHAPES = [s for s in SHAPES if s not in ("triangle", "circle")]


def _extent(o):
    b = poly(o).bounds
    return o["y"] - b[1], b[3] - o["y"]


def equal_ink_partner(rng, shape1, shape2, size_range, fill, stroke):
    s1 = _u(rng, size_range)
    a = ink_area(make_obj(shape1, s1, fill, .5, .5, 0, stroke))
    s2 = size_for_ink(shape2, a, fill, stroke)
    if not size_range[0] <= s2 <= size_range[1]:
        return None
    return s1, s2


def _b1(rng, st):
    stroke = sample_stroke(rng)
    fill = bool(rng.random() < 0.5)
    sz = equal_ink_partner(rng, "circle", "triangle", st["size_range"], fill, stroke)
    if sz is None:
        return None
    sc, stri = sz
    T = make_obj("triangle", stri, fill, _u(rng, POS_RANGE), .5, rng.uniform(0, 360), stroke, "t1")
    C = make_obj("circle", sc, fill, _u(rng, POS_RANGE), .5, 0, stroke, "t2")
    (tu, td), (cu, cd) = _extent(T), _extent(C)
    hT, hC = tu + td, cu + cd
    g = _u(rng, st["knob"])
    H = hT + g + hC
    if H > 1 - 2 * EDGE_MARGIN:
        return None
    a = rng.uniform(EDGE_MARGIN, 1 - EDGE_MARGIN - H)
    pos = [moved(T, y=a + tu), moved(C, y=a + hT + g + cu)]
    twn = [moved(T, y=a + hC + g + tu), moved(C, y=a + cu)]
    if not (layout_ok(pos) and layout_ok(twn)):
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


def _place_outside(rng, objs, I, st, tries=200):
    near = st.get("b3_twin_near")
    O = poly(objs[0])
    for _ in range(tries):
        c = moved(I, x=_u(rng, POS_RANGE), y=_u(rng, POS_RANGE))
        if not in_panel(c):
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
    stroke = sample_stroke(rng)
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


# ---------- B4: two shapes touch; twin separates them by an ink gap >= 2 px at 224 px
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


def _b4(rng, st):
    stroke = sample_stroke(rng)
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
    Bp = moved(B, x=c1[0] + tc * u[0], y=c1[1] + tc * u[1])
    g = _u(rng, st["knob"])
    w = S.stroke_w(A)
    tg = _gap_t(P1, B, c1, u, tc, g + w)
    Bt = moved(B, x=c1[0] + tg * u[0], y=c1[1] + tg * u[1])
    if not (in_panel(A) and in_panel(Bp) and in_panel(Bt)):
        return None
    n = _n(rng, st)
    objs = add_distractors(rng, [A, Bp], n - 2, SHAPES, st["size_range"], stroke)
    if objs is None:
        return None
    if any(poly(d).distance(poly(Bt)) < MIN_GAP for d in objs[2:]):
        return None
    pos = index(_clone(objs))
    twin = index(_clone([A, Bt] + objs[2:]))
    if not (layout_ok(pos, exempt=[(0, 1)]) and layout_ok(twin, exempt=[(0, 1)])):
        return None
    gap_true = ink_gap({"objects": twin}, 0, 1)
    return pos, twin, gap_true, {(1, "x"), (1, "y")}, {}


# ---------- C1: exactly 3 shapes; twin adds a 4th (equal total ink or equal item size)
def _scale_all(objs, k):
    return [finalize(dict(o, size=o["size"] * k)) for o in objs]


def _c1(rng, st, control):
    stroke = sample_stroke(rng)
    sr = st["size_range"]
    s = _u(rng, sr)
    fills = [False] * 3 if st.get("c1_fill") == "outline" else None
    objs = []
    for k in range(3):
        f = None if fills is None else fills[k]
        sh = SHAPES[int(rng.integers(len(SHAPES)))]
        o = make_obj(sh, s, bool(rng.random() < .5) if f is None else f, _u(rng, POS_RANGE), _u(rng, POS_RANGE),
                     rng.uniform(0, 360), stroke, "distractor", mirror=bool(rng.random() < .5))
        objs.append(o)
    if not layout_ok(objs):
        return None
    within = st.get("c1_within_bbox", False)
    box = np.array([min(o["bbox"][0] for o in objs), min(o["bbox"][1] for o in objs),
                    max(o["bbox"][2] for o in objs), max(o["bbox"][3] for o in objs)])
    new = None
    for _ in range(60):
        sh = SHAPES[int(rng.integers(len(SHAPES)))]
        f = False if fills is not None else bool(rng.random() < .5)
        d = make_obj(sh, s, f, _u(rng, POS_RANGE), _u(rng, POS_RANGE), rng.uniform(0, 360), stroke,
                     mirror=bool(rng.random() < .5))
        if within:
            b = d["bbox"]
            if b[0] < box[0] or b[1] < box[1] or b[2] > box[2] or b[3] > box[3]:
                continue
        if in_panel(d) and all(gap_ok(d, o) for o in objs):
            new = d
            break
    if new is None:
        return None
    pos = index(_clone(objs))
    twin = _clone(objs) + [new]
    declared = {(3, "*")}
    if control == "equal_ink":
        target = sum(ink_area(o) for o in pos)
        lo, hi = 0.5, 1.0
        for _ in range(40):
            m = 0.5 * (lo + hi)
            tot = sum(ink_area(o) for o in _scale_all(twin, m))
            lo, hi = (m, hi) if tot < target else (lo, m)
        twin = _scale_all(twin, 0.5 * (lo + hi))
        if twin[0]["size"] < sr[0]:
            return None
        declared |= {(i, "size") for i in range(3)}
    twin = index(twin)
    if not layout_ok(twin):
        return None
    return pos, twin, 3, declared, {"control": control}


SAMPLERS = {"A1": _a1, "A6": _a6, "B1": _b1, "B3": _b3, "B4": _b4}


# ---------- field-by-field twin checker
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
            if same and (i, f) in declared and f != "size":
                v.append(f"declared field unchanged obj{i}.{f}")
    if not RULE_FN[rule](pos):
        v.append("positive fails rule")
    if RULE_FN[rule](twin):
        v.append("twin still satisfies rule")
    check_ink = rule in ("A6", "B1", "B3", "B4") or (rule == "C1" and control == "equal_ink")
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
