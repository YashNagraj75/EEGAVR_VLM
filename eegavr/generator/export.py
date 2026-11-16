import hashlib
import json
import os
import numpy as np
from PIL import Image
from .scene import poly, CHIRAL
from .render import render, render_mask
from .version import GENERATOR_VERSION
from .predicates import nest_pairs, touch_pairs

NAMES = {"circle": "circle", "square": "square", "triangle": "triangle", "pentagon": "pentagon", "cross": "cross",
         "L": "L-shape", "right_trapezoid": "right-angled trapezoid", "irregular_pentagon": "irregular pentagon"}
OBJ_KEYS = ("idx", "shape", "chiral", "symmetry_order", "size", "radius", "fill", "x", "y", "rotation", "mirror",
            "stroke_px", "bbox", "role")
COLLINEAR_TOL = 0.01


def _r(v, d=6):
    return round(float(v), d)


def scene_json(sc, pid, seed, regen):
    obs = []
    for o in sc["objects"]:
        e = {k: o[k] for k in OBJ_KEYS}
        for k in ("size", "radius", "x", "y", "rotation", "stroke_px"):
            e[k] = _r(e[k])
        obs.append(e)
    return {"panel_id": pid, "seed": int(seed), "generator_version": GENERATOR_VERSION,
            "render_style": sc["render_style"], "objects": obs, "truth": {k: bool(v) for k, v in sc["truth"].items()},
            "regen": regen}


def _order(pid, n):
    rng = np.random.default_rng(int(hashlib.sha1(pid.encode()).hexdigest()[:12], 16))
    return list(rng.permutation(n))


def relations(sc, order):
    # symbolic relations over display ids (1-based in the random object order)
    ob = sc["objects"]
    did = {idx: k + 1 for k, idx in enumerate(order)}
    R = []
    for i in order:
        for j in order:
            if i == j:
                continue
            a, b = ob[i]["bbox"], ob[j]["bbox"]
            if a[3] < b[1]:
                R.append(["above", did[i], did[j]])
            if a[2] < b[0]:
                R.append(["left_of", did[i], did[j]])
    for o, i in nest_pairs(sc):
        R.append(["inside", did[i], did[o]])
    for i, j in touch_pairs(sc):
        if not any({i, j} == {a, b} for a, b in nest_pairs(sc)):
            R.append(["touching", *sorted((did[i], did[j]))])
    P = np.array([[o["x"], o["y"]] for o in ob])
    n = len(ob)
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                tri = (a, b, c)
                dev = []
                for m in tri:
                    e = [t for t in tri if t != m]
                    u = P[e[1]] - P[e[0]]
                    dev.append(abs(np.cross(u, P[m] - P[e[0]])) / (np.linalg.norm(u) + 1e-12))
                if max(dev) <= COLLINEAR_TOL:
                    R.append(["collinear", *sorted(did[t] for t in tri)])
                    for m in tri:
                        e = [t for t in tri if t != m]
                        u = P[e[1]] - P[e[0]]
                        s = np.dot(P[m] - P[e[0]], u) / np.dot(u, u)
                        if 0 < s < 1:
                            R.append(["between", did[m], *sorted(did[t] for t in e)])
    for a in range(n):
        for b in range(a + 1, n):
            if ob[a]["shape"] == ob[b]["shape"]:
                R.append(["same_shape", *sorted((did[a], did[b]))])
    sc.pop("_polys", None)
    return R


def described(sc, order):
    # fixed template; never mentions roles or rule names
    parts = []
    for k, i in enumerate(order):
        o = sc["objects"][i]
        f = "filled" if o["fill"] else "outline"
        art = "an" if f == "outline" else "a"
        s = (f"Shape {k + 1}: {art} {f} {NAMES[o['shape']]}, centre at x={o['x']:.3f} from the left, "
             f"y={o['y']:.3f} from the top, size {o['radius']:.3f}")
        if o["shape"] != "circle":
            s += f", rotated {o['rotation']:.0f} degrees"
        if o["mirror"]:
            s += ", mirrored"
        parts.append(s + ".")
    return " ".join(parts)


def symbolic(sc, order):
    objs = []
    for k, i in enumerate(order):
        o = sc["objects"][i]
        e = {"id": k + 1, "shape": o["shape"], "fill": "filled" if o["fill"] else "outline", "x": _r(o["x"], 4),
             "y": _r(o["y"], 4), "size": _r(o["radius"], 4), "rotation": _r(o["rotation"], 1)}
        if CHIRAL[o["shape"]]:
            e["mirrored"] = bool(o["mirror"])
        objs.append(e)
    return {"objects": objs, "relations": relations(sc, order)}


def write_panel(root, pid, sc_json, sc):
    img = render(sc)
    m = render_mask(sc)
    Image.fromarray(img).save(f"{root}/panels/{pid}.png", optimize=True)
    Image.fromarray(m).save(f"{root}/masks/{pid}.png", optimize=True)
    with open(f"{root}/scenes/{pid}.json", "w") as f:
        json.dump(sc_json, f)


def ensure_dirs(root):
    for d in ("panels", "masks", "scenes", "problems", "splits", "audit"):
        os.makedirs(f"{root}/{d}", exist_ok=True)
