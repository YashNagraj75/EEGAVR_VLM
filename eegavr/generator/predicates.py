from .scene import poly, stroke_w, CONCAVE

PILOT_RULES = ("A1", "A6", "B1", "B3", "B4", "C1", "C1b", "A2")
TOUCH_EPS = 1e-6


def _geom(scene):
    if "_polys" not in scene:
        scene["_polys"] = [poly(o) for o in scene["objects"]]
    return scene["_polys"]


def a1(scene, min_size=0.0, max_size=1.0):
    return any(o["fill"] and min_size <= o["size"] <= max_size for o in scene["objects"])


def a6(scene):
    return any(o["fill"] and o["shape"] == "triangle" for o in scene["objects"])


def b1(scene, gap=0.0):
    # some triangle's lowest point is higher (smaller y) than some circle's highest point, by > gap
    ob = scene["objects"]
    return any(c["bbox"][1] - t["bbox"][3] > gap for t in ob if t["shape"] == "triangle"
               for c in ob if c["shape"] == "circle")


def b2(scene, gap=0.0):
    ob = scene["objects"]
    return any(c["bbox"][0] - t["bbox"][2] > gap for t in ob if t["shape"] == "triangle"
               for c in ob if c["shape"] == "circle")


def nest_pairs(scene, margin=0.0):
    ps = _geom(scene)
    out = []
    for i, P in enumerate(ps):
        for j, Q in enumerate(ps):
            if i != j and P.contains(Q) and P.exterior.distance(Q) >= margin:
                out.append((i, j))
    return out


def b3(scene, margin=0.0):
    return len(nest_pairs(scene, margin)) > 0


def ink_gap(scene, i, j):
    ps = _geom(scene)
    w = 0.5 * (stroke_w(scene["objects"][i]) + stroke_w(scene["objects"][j]))
    return ps[i].distance(ps[j]) - w


def touch_pairs(scene, tol=0.0):
    # boundaries meet: drawn strokes touch (ink gap <= tol), filled regions do not overlap, no nesting
    ps = _geom(scene)
    out = []
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            if ps[i].contains(ps[j]) or ps[j].contains(ps[i]):
                continue
            if ps[i].intersection(ps[j]).area > TOUCH_EPS * 1e-2:
                continue
            if ink_gap(scene, i, j) <= tol:
                out.append((i, j))
    return out


def b4(scene, tol=0.0):
    return len(touch_pairs(scene, tol)) > 0 and not b3(scene)


def c1(scene, k=3):
    return len(scene["objects"]) == k


def c1b(scene, k=3):
    return sum(o["shape"] == "triangle" for o in scene["objects"]) == k


def a2(scene):
    return any(o["shape"] == "triangle" for o in scene["objects"])


def truth(scene):
    return {"A1": a1(scene), "A6": a6(scene), "B1": b1(scene), "B3": b3(scene), "B4": b4(scene), "C1": c1(scene),
            "C1b": c1b(scene), "A2": a2(scene)}


RULE_FN = {"A1": a1, "A6": a6, "B1": b1, "B3": b3, "B4": b4, "C1": c1, "C1b": c1b, "A2": a2}


def candidate_predicates(extended=True):
    # every predicate the 5c check can express: pilot rules at every knob threshold plus library rules
    C = {}
    for t in (0.06, 0.08, 0.09, 0.10, 0.12, 0.14):
        C[f"A1:filled size>={t}"] = lambda s, t=t: a1(s, min_size=t)
        C[f"A1:filled size<={t}"] = lambda s, t=t: a1(s, max_size=t)
    C["A1:some filled"] = a1
    C["A6:filled triangle"] = a6
    for t in (0.0, 0.03, 0.05, 0.08, 0.12, 0.16, 0.20):
        C[f"B1:triangle above circle gap>{t}"] = lambda s, t=t: b1(s, t)
        C[f"B1':circle above triangle gap>{t}"] = lambda s, t=t: _b1_rev(s, t)
    for t in (0.0, 0.03, 0.045, 0.06, 0.08):
        C[f"B3:inside margin>={t}"] = lambda s, t=t: b3(s, t)
    for t in (0.0, 0.01, 0.02, 0.04, 0.07, 0.10):
        C[f"B4:ink gap<={t}"] = lambda s, t=t: len(touch_pairs(s, t)) > 0
    for k in range(1, 10):
        C[f"C:count=={k}"] = lambda s, k=k: len(s["objects"]) == k
        C[f"C:count>={k}"] = lambda s, k=k: len(s["objects"]) >= k
    C["A2:triangle present"] = a2
    for k in range(0, 6):
        C[f"C1b:triangle count=={k}"] = lambda s, k=k: c1b(s, k)
        C[f"C1b:triangle count>={k}"] = lambda s, k=k: sum(o["shape"] == "triangle" for o in s["objects"]) >= k
    for t in (0.10, 0.12, 0.14):
        C[f"A3:size>={t} present"] = lambda s, t=t: any(o["size"] >= t for o in s["objects"])
    C["A5:concave present"] = lambda s: any(o["shape"] in CONCAVE for o in s["objects"])
    C["B2:triangle left of circle"] = b2
    for k in range(0, 5):
        C[f"C4:filled count=={k}"] = lambda s, k=k: sum(o["fill"] for o in s["objects"]) == k
        C[f"C4:filled count>={k}"] = lambda s, k=k: sum(o["fill"] for o in s["objects"]) >= k
    if extended:
        from .scene import SHAPES
        for sh in SHAPES:
            for f in (True, False):
                C[f"X:{'filled' if f else 'outline'} {sh} present"] = (
                    lambda s, sh=sh, f=f: any(o["shape"] == sh and o["fill"] == f for o in s["objects"]))
    return C


def _b1_rev(scene, gap=0.0):
    ob = scene["objects"]
    return any(t["bbox"][1] - c["bbox"][3] > gap for t in ob if t["shape"] == "triangle"
               for c in ob if c["shape"] == "circle")
