import math
import numpy as np
from shapely.geometry import Polygon
from shapely import affinity

PANEL_PX = 512
SIZE_RANGE = (0.06, 0.16)
POS_RANGE = (0.10, 0.90)
MIN_GAP = 0.03
EDGE_MARGIN = 0.02
STROKE_BASE_PX = 4.0
STROKE_JITTER = 0.25


def _regular(n, start_deg=90.0):
    a = np.deg2rad(start_deg + 360.0 * np.arange(n) / n)
    return np.c_[np.cos(a), np.sin(a)]


_RAW = {
    "circle": _regular(96),
    "square": np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float),
    "triangle": _regular(3),
    "pentagon": _regular(5),
    "cross": np.array([[-1/3, -1], [1/3, -1], [1/3, -1/3], [1, -1/3], [1, 1/3], [1/3, 1/3], [1/3, 1],
                       [-1/3, 1], [-1/3, 1/3], [-1, 1/3], [-1, -1/3], [-1/3, -1/3]], float),
    "L": np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 3], [0, 3]], float),
    "right_trapezoid": np.array([[0, 0], [2.2, 0], [1.2, 1.4], [0, 1.4]], float),
    "irregular_pentagon": np.array([[0, 0], [1.7, 0.1], [2.1, 1.1], [1.0, 2.0], [-0.3, 1.2]], float),
}
SYMMETRY = {"circle": 0, "square": 4, "triangle": 3, "pentagon": 5, "cross": 4,
            "L": 1, "right_trapezoid": 1, "irregular_pentagon": 1}
CHIRAL = {k: k in ("L", "right_trapezoid", "irregular_pentagon") for k in _RAW}
SHAPES = list(_RAW)
CONCAVE = ("cross", "L")


def _normalise(v):
    # centre on the area centroid and scale so the outline perimeter is 2*pi
    p = Polygon(v)
    c = np.array(p.centroid.coords[0])
    v = v - c
    return v * (2 * math.pi / Polygon(v).length)


UNIT = {k: _normalise(v) for k, v in _RAW.items()}
UNIT_R = {k: float(np.max(np.hypot(*UNIT[k].T))) for k in UNIT}
UNIT_AREA = {k: Polygon(UNIT[k]).area for k in UNIT}


def poly(o):
    # filled region in panel coords (x right, y down); rotation is visual CCW
    v = UNIT[o["shape"]].copy()
    if o.get("mirror"):
        v[:, 0] *= -1
    t = np.deg2rad(o["rotation"])
    c, s = math.cos(t), math.sin(t)
    u = v[:, 0] * c - v[:, 1] * s
    w = v[:, 0] * s + v[:, 1] * c
    return Polygon(np.c_[o["x"] + o["size"] * u, o["y"] - o["size"] * w])


def stroke_w(o):
    return o["stroke_px"] / PANEL_PX


def ink_geom(o, style="default"):
    # exact drawn ink: filled = polygon grown by half a stroke; outline = ring of stroke width
    p, w = poly(o), stroke_w(o)
    if o["fill"]:
        return p.buffer(w / 2, join_style="mitre", mitre_limit=3.0)
    return p.exterior.buffer(w / 2, join_style="mitre", mitre_limit=3.0)


def ink_area(o):
    return ink_geom(o).area


def finalize(o):
    p = poly(o)
    o["radius"] = round(float(o["size"] * UNIT_R[o["shape"]]), 6)
    o["bbox"] = [round(float(b), 6) for b in p.bounds]
    o["chiral"] = CHIRAL[o["shape"]]
    o["symmetry_order"] = SYMMETRY[o["shape"]]
    return o


def make_obj(shape, size, fill, x, y, rotation, stroke_px, role="distractor", mirror=False):
    o = {"idx": -1, "shape": shape, "chiral": CHIRAL[shape], "symmetry_order": SYMMETRY[shape],
         "size": float(size), "radius": 0.0, "fill": bool(fill), "x": float(x), "y": float(y),
         "rotation": float(rotation % 360.0) if shape != "circle" else 0.0,
         "mirror": bool(mirror and CHIRAL[shape]), "stroke_px": float(stroke_px), "bbox": None, "role": role}
    return finalize(o)


def moved(o, x=None, y=None):
    n = dict(o)
    if x is not None:
        n["x"] = float(x)
    if y is not None:
        n["y"] = float(y)
    return finalize(n)


HATCH_STROKE = 1.5


def sample_stroke(rng, style="default"):
    # per-panel stroke width in px (the drawn width; the hatch style draws 1.5x wider strokes)
    k = HATCH_STROKE if style == "hatch" else 1.0
    return float(k * STROKE_BASE_PX * rng.uniform(1 - STROKE_JITTER, 1 + STROKE_JITTER))


def random_obj(rng, shapes, size_range, stroke_px, fill=None, role="distractor"):
    sh = shapes[int(rng.integers(len(shapes)))]
    f = bool(rng.random() < 0.5) if fill is None else fill
    return make_obj(sh, rng.uniform(*size_range), f, rng.uniform(*POS_RANGE), rng.uniform(*POS_RANGE),
                    rng.uniform(0, 360), stroke_px, role, mirror=bool(rng.random() < 0.5))


def in_panel(o):
    b = o["bbox"]
    return (b[0] >= EDGE_MARGIN and b[1] >= EDGE_MARGIN and b[2] <= 1 - EDGE_MARGIN and b[3] <= 1 - EDGE_MARGIN
            and POS_RANGE[0] <= o["x"] <= POS_RANGE[1] and POS_RANGE[0] <= o["y"] <= POS_RANGE[1])


def gap_ok(a, b, gap=MIN_GAP):
    return poly(a).distance(poly(b)) >= gap


def layout_ok(objs, exempt=()):
    # every object inside the panel; every pair >= MIN_GAP apart except exempt index pairs
    if not all(in_panel(o) for o in objs):
        return False
    ps = [poly(o) for o in objs]
    ex = {frozenset(e) for e in exempt}
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            if frozenset((i, j)) in ex:
                continue
            if ps[i].distance(ps[j]) < MIN_GAP:
                return False
    return True


def add_distractors(rng, objs, n, shapes, size_range, stroke_px, tries=60, fills=None):
    # place n random distractors with the minimum gap; returns None if they do not fit
    out = list(objs)
    for k in range(n):
        for _ in range(tries):
            f = None if fills is None else fills[k]
            sh = shapes if not isinstance(shapes[0], (list, tuple)) else shapes[k]
            d = random_obj(rng, sh, size_range, stroke_px, fill=f)
            if in_panel(d) and all(gap_ok(d, o) for o in out):
                out.append(d)
                break
        else:
            return None
    return out


def index(objs):
    for i, o in enumerate(objs):
        o["idx"] = i
    return objs


def size_for_ink(shape, target_ink, fill, stroke_px, lo=0.02, hi=0.40):
    # bisection on size so that the object's drawn ink equals target_ink
    for _ in range(40):
        m = 0.5 * (lo + hi)
        a = ink_area(make_obj(shape, m, fill, 0.5, 0.5, 0.0, stroke_px))
        lo, hi = (m, hi) if a < target_ink else (lo, m)
    return 0.5 * (lo + hi)
