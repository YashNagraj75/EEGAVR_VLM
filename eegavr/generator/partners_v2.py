import numpy as np
from . import rules_a as RA
from .render import object_ink
from .scene import finalize, layout_ok, gap_ok, in_panel, poly, CHIRAL, CONCAVE, SYMMETRY
from .predicates import RULE_FN
from .rules_borient import CONFUSABLE

V20_PARTNERS = {"B2": "A2", "B5": "A3", "B6": "A4", "B7": "A5", "B8": "A2", "B9": "A3"}
TRIES = 40
ROT_TRIES = 12
GAP_TOL = 1e-9
B6_OFF = (3.0, 7.0)
ROLE_RESID = (0.0, 2.0, -2.0)
A4_TARGETS = ("cross", "square", "triangle", "pentagon")
A5_TWIN = ("square", "pentagon")
B8_SUBST = ("square", "cross", "L", "right_trapezoid", "circle", "irregular_pentagon")


def _clone(objs):
    return [dict(o) for o in objs]


def _set(objs, j, **kw):
    out = _clone(objs)
    out[j] = finalize(dict(out[j], **kw))
    return out


def _roles(objs):
    for o in objs:
        o["role"] = "distractor"
    return objs


def _fits(objs, k):
    return in_panel(objs[k]) and all(gap_ok(objs[k], o) for i, o in enumerate(objs) if i != k)


def _arrangement_ok(pos, twin, j):
    # both panels valid, the target's drawn ink >= 2 px from the edge, scene minimum gap unchanged, difference >= 2 px
    m = RA.MIN_DIFF_PX / RA.MIN_INPUT_PX
    clear = all(m <= b[0] and m <= b[1] and b[2] <= 1 - m and b[3] <= 1 - m
                for b in (object_ink(o).bounds for o in pos + [twin[j]]))
    pair = {"pos": {"objects": pos}, "twin": {"objects": twin}, "meta": {"target": j}}
    return (clear and layout_ok(pos) and layout_ok(twin) and abs(RA.min_gap(pos) - RA.min_gap(twin)) < GAP_TOL
            and RA.rule_diff(pair, RA.MIN_INPUT_PX) >= RA.MIN_DIFF_PX)


def _swap_shape(rng, objs, j, shape, **kw):
    # replacement at a rotation keeping j's nearest-neighbour gap, else one keeping j out of the closest pair
    rot = RA.matched_rotation(rng, objs, j, shape, False)
    if rot is not None:
        return _set(objs, j, shape=shape, mirror=False, rotation=rot, **kw)
    if not RA._not_closest(objs, j):
        return None
    for r in rng.uniform(0, 360, ROT_TRIES):
        out = _set(objs, j, shape=shape, mirror=False, rotation=float(r), **kw)
        if layout_ok(out) and RA._not_closest(out, j):
            return out
    return None


def _to_pentagon(rng, objs, j):
    return _swap_shape(rng, objs, j, "pentagon", fill=False)


def _a2_b2(rng, objs, state):
    # target = the B2 triangle role; twin: same-size outline pentagon at a gap-preserving rotation
    j = next(k for k, o in enumerate(objs) if o["shape"] == "triangle")
    d = [k for k, o in enumerate(objs) if o["role"] == "distractor"]
    objs = _roles(_clone(objs))
    objs[j]["role"] = "t1"
    has_pent = any(o["shape"] in ("pentagon", "irregular_pentagon") for o in objs)
    if d and not has_pent and rng.random() < 0.5:
        objs = _to_pentagon(rng, objs, d[int(rng.integers(len(d)))])
        if objs is None:
            return None
    for r in rng.uniform(0, 360, ROT_TRIES):
        pos = _set(objs, j, rotation=float(r))
        if layout_ok(pos) and RULE_FN["B2"]({"objects": pos}) == state:
            twin = _to_pentagon(rng, pos, j)
            if twin is not None:
                return pos, twin, j, float(len(pos) - 1)
    return None


def _a2_b8(rng, objs, state):
    # B8 roles kept (triangles recoded; outline roles may become pentagons); the distractor is the triangle target
    objs = _clone(objs)
    S = next(o["shape"] for o in objs if o["role"] == "t2")
    O = next(o["shape"] for o in objs if o["role"] == "t1")
    for role, sh, other in (("t2", S, O), ("t1", O, S)):
        if sh == "triangle":
            ok = [s for s in B8_SUBST if s not in (other, CONFUSABLE.get(other)) and s not in [o["shape"] for o in objs]]
            new = ok[int(rng.integers(len(ok)))]
            mir = bool(rng.random() < 0.5) and CHIRAL[new]
            for k, o in enumerate(objs):
                if o["role"] == role:
                    objs = _set(objs, k, shape=new, mirror=mir)
    shapes = {o["shape"] for o in objs}
    if not next(o["fill"] for o in objs if o["role"] == "t2") and not shapes & {"pentagon", "irregular_pentagon"}:
        role = ("t1", "t2")[int(rng.integers(2))]
        for k, o in enumerate(objs):
            if o["role"] == role:
                objs = _set(objs, k, shape="pentagon", mirror=False)
    d = [k for k, o in enumerate(objs) if o["role"] == "distractor"]
    if not d or not layout_ok(objs):
        return None
    j = d[int(rng.integers(len(d)))]
    objs = _roles(objs)
    for r in rng.uniform(0, 360, ROT_TRIES):
        pos = _set(objs, j, shape="triangle", mirror=False, rotation=float(r), fill=False, role="t1")
        if layout_ok(pos):
            twin = _to_pentagon(rng, pos, j)
            if twin is not None:
                return pos, twin, j, float(len(pos) - 1)
    return None


def _a3_b(rng, objs, state):
    # target at 0.11 + g, all other shapes U[0.06, 0.11 - g]; twin shrinks the target; positions kept
    T, g = RA.A3_THRESHOLD, float(rng.uniform(*RA.FULL_A["A3"]))
    order = sorted(range(len(objs)), key=lambda k: -RA.nn_gap(objs, k) * rng.uniform(0.8, 1.0))
    for j in order:
        pos = _roles(_clone(objs))
        for k in range(len(pos)):
            pos = _set(pos, k, size=T + g if k == j else float(rng.uniform(0.06, T - g)))
        pos[j]["role"] = "t1"
        twin = _set(pos, j, size=T - g)
        if layout_ok(pos) and RA._not_closest(pos, j) and RA._not_closest(twin, j):
            return pos, twin, j, g
    return None


def _to_upright(o):
    # smallest rotation bringing o to an upright pose (a multiple of its symmetry period)
    p = 360.0 / SYMMETRY[o["shape"]]
    r = o["rotation"] % p
    return -r if r <= p / 2 else p - r


def _signed_diff(a, b):
    # signed difference of b's rotation from a's, folded into (-P/2, P/2]
    p = 360.0 / SYMMETRY[a["shape"]]
    d = (b["rotation"] - a["rotation"]) % p
    return d if d <= p / 2 else d - p


def _upright_scene(rng, objs, roles, state):
    # every non-circular shape within 2 deg of upright (roles share one residual, so B6 holds iff state); None if crowded
    t1, t2 = objs[roles[0]], objs[roles[1]]
    off = 0.0 if state else float(rng.uniform(*B6_OFF)) * (1 if rng.random() < 0.5 else -1)
    for e in ROLE_RESID:
        up = _roles(_clone(objs))
        up = _set(up, roles[0], rotation=t1["rotation"] + _to_upright(t1) + e)
        up = _set(up, roles[1], rotation=up[roles[0]]["rotation"] + off + (t2["rotation"] - t1["rotation"]
                                                                           - _signed_diff(t1, t2)))
        if in_panel(up[roles[0]]) and in_panel(up[roles[1]]) and gap_ok(up[roles[0]], up[roles[1]]):
            break
    else:
        return None
    for k, o in enumerate(objs):
        if k in roles or o["shape"] == "circle":
            continue
        for e in ROLE_RESID:
            up = _set(up, k, rotation=o["rotation"] + _to_upright(o) + e)
            if _fits(up, k):
                break
        else:
            up = _set(up, k, shape="circle", rotation=0.0, mirror=False)
    return up if layout_ok(up) else None


def _a4_b6(rng, objs, state):
    # one distractor tilted 45 - knob (or the gap-preserving knob), every other shape near upright; twin: upright
    roles = [k for k, o in enumerate(objs) if o["role"] in ("t1", "t2")]
    up = _upright_scene(rng, objs, roles, state)
    used = {o["shape"] for k, o in enumerate(objs) if k not in roles}
    for sh in rng.permutation([s for s in A4_TARGETS if s not in used]):
        if up is not None:
            break
        alt = _clone(objs)
        for k in roles:
            alt = _set(alt, k, shape=str(sh), mirror=False)
        up = _upright_scene(rng, alt, roles, state)
    if up is None:
        return None
    for j in rng.permutation([k for k in range(len(up)) if k not in roles]):
        j = int(j)
        base = up
        if base[j]["shape"] == "circle" or not _fits(base, j):
            used = [o["shape"] for o in base]
            alts = [_set(base, j, shape=str(sh), rotation=0.0, mirror=False) for sh in rng.permutation(A4_TARGETS)
                    if sh not in used]
            alts = [x for x in alts if _fits(x, j)]
            if not alts:
                continue
            base = alts[0]
        per = 360.0 / SYMMETRY[base[j]["shape"]]
        lo = max(RA.FULL_A["A4"][0], 45.0 - per / 2)
        if lo >= RA.FULL_A["A4"][1]:
            continue
        sign = 1 if rng.random() < 0.5 else -1
        twin = _set(base, j, rotation=base[j]["rotation"] - (base[j]["rotation"] % per if
                                                              base[j]["rotation"] % per <= per / 2 else
                                                              base[j]["rotation"] % per - per), role="t1")
        delta = float(rng.uniform(lo, RA.FULL_A["A4"][1]))
        pos = _set(twin, j, rotation=twin[j]["rotation"] + sign * (45.0 - delta))
        if not (RA._not_closest(pos, j) and RA._not_closest(twin, j)):
            delta = _gap_tilt(twin, j, sign, lo)
            if delta is None:
                continue
            pos = _set(twin, j, rotation=twin[j]["rotation"] + sign * (45.0 - delta))
        if layout_ok(pos) and layout_ok(twin):
            return pos, twin, j, delta
    return None


def _gap_tilt(twin, j, sign, lo):
    # knob in [lo, max] whose tilt keeps the target's nearest-neighbour gap equal to the upright twin's
    target = RA.nn_gap(twin, j)
    others = [poly(o) for i, o in enumerate(twin) if i != j]
    up = twin[j]["rotation"]

    def f(d):
        return min(poly(dict(twin[j], rotation=up + sign * (45.0 - d))).distance(q) for q in others) - target

    grid = np.linspace(lo, RA.FULL_A["A4"][1], 61)
    v = np.array([f(d) for d in grid])
    idx = np.nonzero(np.sign(v[:-1]) != np.sign(v[1:]))[0]
    if len(idx) == 0:
        return None
    a, b, fa = grid[idx[0]], grid[idx[0] + 1], v[idx[0]]
    for _ in range(40):
        m = 0.5 * (a + b)
        fm = f(m)
        a, b, fa = (m, b, fm) if np.sign(fm) == np.sign(fa) else (a, m, fa)
    return float(0.5 * (a + b))


def _a5_b7(rng, objs, state):
    # L roles become another chiral shape; one distractor is a cross or L, the others and the twin share one convex shape
    roles = [k for k, o in enumerate(objs) if o["role"] in ("t1", "t2")]
    objs = _roles(_clone(objs))
    sh = objs[roles[0]]["shape"]
    if sh == "L":
        sh = "irregular_pentagon" if objs[roles[0]]["size"] >= 0.09 and rng.random() < 0.5 else "right_trapezoid"
        for k in roles:
            objs = _set(objs, k, shape=sh)
    cand = [k for k in range(len(objs)) if k not in roles]
    for k in cand:
        if objs[k]["shape"] in CONCAVE or objs[k]["shape"] in (sh, CONFUSABLE.get(sh)):
            objs = _set(objs, k, shape="square", mirror=False)
    if not cand:
        return None
    j = cand[int(rng.integers(len(cand)))]
    new = [s for s in A5_TWIN if s != CONFUSABLE.get(sh)]
    new = new[int(rng.integers(len(new)))]
    for k in cand:
        if k != j:
            objs = _set(objs, k, shape=new, mirror=False, fill=False)
    conc = CONCAVE[int(rng.integers(len(CONCAVE)))]
    d = float(rng.uniform(*RA.FULL_A["A5"]))
    size = d / RA.UNIT_DEPTH[conc]
    if not 0.06 <= size <= 0.16:
        return None
    pos = _set(objs, j, shape=conc, size=size, rotation=float(rng.uniform(0, 360)), fill=False,
               mirror=bool(rng.random() < 0.5), role="t1")
    rot = RA.matched_rotation(rng, pos, j, new, False)
    if rot is None:
        return None
    twin = _set(pos, j, shape=new, mirror=False, rotation=rot)
    if abs(RA._ink(twin) - RA._ink(pos)) / RA._ink(pos) > RA.INK_TOL:
        return None
    return pos, twin, j, d


BUILD = {"B2": _a2_b2, "B5": _a3_b, "B6": _a4_b6, "B7": _a5_b7, "B8": _a2_b8, "B9": _a3_b}


def build_v2(b_rule, b_pair, state, salt):
    # one A-partner pair from a B base scene: positions and B truth kept; None if no valid pair in TRIES draws
    base = b_pair["pos"] if state else b_pair["twin"]
    rng = np.random.default_rng([b_pair["seed"], 13, int(state), int(salt)])
    want = RULE_FN[b_rule](base)
    for _ in range(TRIES):
        r = BUILD[b_rule](rng, base["objects"], state)
        if r is None:
            continue
        pos, twin, j, knob = r
        if not _arrangement_ok(pos, twin, j):
            continue
        if RULE_FN[b_rule]({"objects": pos}) != want or RULE_FN[b_rule]({"objects": twin}) != (want and b_rule != "B2"):
            continue
        return pos, twin, j, knob
    return None
