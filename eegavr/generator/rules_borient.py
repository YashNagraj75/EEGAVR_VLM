import math
import numpy as np
from .scene import (make_obj, finalize, layout_ok, add_distractors, index, sample_stroke, SHAPES, POS_RANGE, UNIT_R,
                    SYMMETRY, CHIRAL)

ORIENTED = [s for s in SHAPES if SYMMETRY[s] != 0]
CHIRAL_SHAPES = [s for s in SHAPES if CHIRAL[s]]
SAME_TOL_DEG = 1.0
CHECK_PX = 224
MIN_DIFF_PX = 2.0
CONFUSABLE = {"pentagon": "irregular_pentagon", "irregular_pentagon": "pentagon"}
CHIRAL_HAUSDORFF = {"L": 0.20, "right_trapezoid": 0.25, "irregular_pentagon": 0.10}


def period(shape):
    return 360.0 / SYMMETRY[shape]


def angle_diff(a, b):
    # orientation difference folded modulo the shared symmetry period, in [0, period/2]
    p = period(a["shape"])
    d = (a["rotation"] - b["rotation"]) % p
    return min(d, p - d)


def orientation_pairs(scene, tol=SAME_TOL_DEG):
    # same non-circular shape (and handedness), orientations equal modulo symmetry within tol degrees
    ob = scene["objects"]
    out = []
    for i in range(len(ob)):
        for j in range(i + 1, len(ob)):
            a, b = ob[i], ob[j]
            if a["shape"] != b["shape"] or SYMMETRY[a["shape"]] == 0 or a.get("mirror") != b.get("mirror"):
                continue
            if angle_diff(a, b) <= tol:
                out.append((i, j))
    return out


def mirror_pairs(scene):
    ob = scene["objects"]
    return [(i, j) for i in range(len(ob)) for j in range(i + 1, len(ob)) if ob[i]["shape"] == ob[j]["shape"]
            and CHIRAL[ob[i]["shape"]] and bool(ob[i].get("mirror")) != bool(ob[j].get("mirror"))]


def b6(scene, tol=SAME_TOL_DEG):
    return len(orientation_pairs(scene, tol)) > 0


def b7(scene):
    return len(mirror_pairs(scene)) > 0


def rotation_diff_px(o, d, px=CHECK_PX):
    # largest vertex displacement (px at the check size) when o is rotated by d degrees
    return o["size"] * UNIT_R[o["shape"]] * 2 * math.sin(math.radians(d) / 2) * px


def chirality_px(o, px=CHECK_PX):
    return CHIRAL_HAUSDORFF[o["shape"]] * o["size"] * px


def _u(rng, r):
    return float(rng.uniform(r[0], r[1]))


def _n(st):
    return int(st["n_objects"][0])


def _distractors(rng, roles, st, role_shape, stroke):
    # distinct distractor shapes, never the role shape or its look-alike, so no second same-shape pair exists
    k = _n(st) - len(roles)
    pool = [s for s in SHAPES if s not in (role_shape, CONFUSABLE.get(role_shape))]
    picks = [pool[i] for i in rng.permutation(len(pool))[:k]]
    if len(picks) < k:
        return None
    return add_distractors(rng, roles, k, [[s] for s in picks], st["size_range"], stroke)


def _role_pair(rng, st, shape, size, fill, stroke, rot1, rot2, mir1, mir2):
    A = make_obj(shape, size, fill, _u(rng, POS_RANGE), _u(rng, POS_RANGE), rot1, stroke, "t1", mirror=mir1)
    B = make_obj(shape, size, fill, _u(rng, POS_RANGE), _u(rng, POS_RANGE), rot2, stroke, "t2", mirror=mir2)
    return A, B


def _b6(rng, st):
    # two copies of a non-circular shape, equal orientation modulo symmetry; twin rotates t2 by +-d
    stroke = sample_stroke(rng, st["style"])
    d = _u(rng, st["knob"])
    shapes = [s for s in ORIENTED if period(s) / 2 >= d]
    sh = shapes[int(rng.integers(len(shapes)))]
    size = _u(rng, st["size_range"])
    fill, mir = bool(rng.random() < 0.5), bool(rng.random() < 0.5)
    th = float(rng.uniform(0, 360))
    off = period(sh) * int(rng.integers(SYMMETRY[sh]))
    A, B = _role_pair(rng, st, sh, size, fill, stroke, th, th + off, mir, mir)
    if rotation_diff_px(A, d) < MIN_DIFF_PX:
        return None
    Bt = finalize(dict(B, rotation=(B["rotation"] + (d if rng.random() < 0.5 else -d)) % 360.0))
    if not (layout_ok([A, B]) and layout_ok([A, Bt])):
        return None
    objs = _distractors(rng, [A, B], st, sh, stroke)
    if objs is None:
        return None
    pos = index([dict(o) for o in objs])
    twin = index([dict(A), Bt] + [dict(o) for o in objs[2:]])
    if not layout_ok(twin):
        return None
    return pos, twin, angle_diff(twin[0], twin[1]), {(1, "rotation")}, {"role_shape": sh}


def _b7(rng, st):
    # positive: t2 = t1 mirrored about the vertical axis, rotated by phi; twin: t2 = t1 rotated by phi
    stroke = sample_stroke(rng, st["style"])
    phi = _u(rng, st["knob"]) * (1 if rng.random() < 0.5 else -1)
    sh = CHIRAL_SHAPES[int(rng.integers(len(CHIRAL_SHAPES)))]
    size = _u(rng, st["size_range"])
    fill, mir = bool(rng.random() < 0.5), bool(rng.random() < 0.5)
    th = float(rng.uniform(0, 360))
    A, B = _role_pair(rng, st, sh, size, fill, stroke, th, -th + phi, mir, not mir)
    if chirality_px(A) < MIN_DIFF_PX:
        return None
    Bt = finalize(dict(B, rotation=(th + phi) % 360.0, mirror=mir))
    if not (layout_ok([A, B]) and layout_ok([A, Bt])):
        return None
    objs = _distractors(rng, [A, B], st, sh, stroke)
    if objs is None:
        return None
    pos = index([dict(o) for o in objs])
    twin = index([dict(A), Bt] + [dict(o) for o in objs[2:]])
    if not layout_ok(twin):
        return None
    return pos, twin, abs(phi), {(1, "rotation"), (1, "mirror")}, {"role_shape": sh}


def borient_candidates():
    # 5c library entries: B6 at several angle tolerances, B7 mirror pair
    C = {f"B6:same orientation tol<={t}": (lambda s, t=t: b6(s, t)) for t in (1.0, 5.0, 10.0, 20.0, 35.0)}
    C["B7:mirror pair"] = b7
    return C


PREDICATES = {"B6": b6, "B7": b7}
SAMPLERS = {"B6": _b6, "B7": _b7}
RULE_TYPE = {"B6": "B", "B7": "B"}
RULE_PROGRAM = {
    "B6": "exists a, b: a.shape == b.shape != circle, same handedness, (a.rot - b.rot) mod (360/a.symmetry) == 0",
    "B7": "exists a, b: a.shape == b.shape, chiral, a.mirror != b.mirror  (roles share size: mirror images)",
}
KNOB_NAME = {"B6": "orientation difference in the twin (degrees, modulo symmetry)",
             "B7": "rotation phi between the two shapes (degrees; vertical-axis mirror in the positive)"}
ID_BINS = {"B6": [(10.0, 20.0), (20.0, 35.0), (35.0, 60.0)], "B7": [(0.0, 60.0), (60.0, 120.0), (120.0, 180.0)]}
FULL = {"B6": (10.0, 60.0), "B7": (0.0, 180.0)}
TUNING = {"B6": {"n_objects_id": (4, 4)}, "B7": {"n_objects_id": (4, 4)}}
FAMILY = {"B6": ("B6:",), "B7": ("B7:",)}
PIX_RELATION = {"B6": ("dir_horizontal", "dir_vertical", "dir_diagonal")}
SYM_DROP = {"B6": ("t1_rot_c", "t1_rot_s"), "B7": ("t1_rot_c", "t1_rot_s")}
