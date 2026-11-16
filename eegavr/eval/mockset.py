import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SHAPES = {"circle": (0, False), "square": (4, False), "triangle": (3, False)}


def _poly(shape, x, y, r, rot):
    n = {"square": 4, "triangle": 3}[shape]
    off = 45 if shape == "square" else 90
    return [(x + r * math.cos(math.radians(rot + off + 360 * k / n)), y - r * math.sin(math.radians(rot + off + 360 * k / n)))
            for k in range(n)]


def render(objects, size=512):
    img, mask = Image.new("L", (size, size), 255), Image.new("L", (size, size), 0)
    d, dm = ImageDraw.Draw(img), ImageDraw.Draw(mask)
    for o in objects:
        x, y, r, w = o["x"] * size, o["y"] * size, o["radius"] * size, max(1, round(o["stroke_px"]))
        fill = 0 if o["fill"] else None
        if o["shape"] == "circle":
            box = [x - r, y - r, x + r, y + r]
            d.ellipse(box, outline=0, width=w, fill=fill)
            dm.ellipse(box, fill=o["idx"] + 1)
        else:
            pts = _poly(o["shape"], x, y, r, o["rotation"])
            d.polygon(pts, outline=0, width=w, fill=fill)
            dm.polygon(pts, fill=o["idx"] + 1)
    return img, mask


def _obj(idx, shape, x, y, r, fill, rng, role="distractor"):
    sym, chiral = SHAPES[shape]
    return {"idx": idx, "shape": shape, "chiral": chiral, "symmetry_order": sym, "radius": r, "fill": fill,
            "x": round(x, 3), "y": round(y, 3), "rotation": float(rng.randrange(0, 360, 5)) if shape != "circle" else 0.0,
            "stroke_px": round(rng.uniform(3.0, 5.0), 2),
            "bbox": [round(x - r, 3), round(y - r, 3), round(x + r, 3), round(y + r, 3)], "role": role}


def _truth(objs):
    tri = [o for o in objs if o["role"] == "t1"]
    cir = [o for o in objs if o["role"] == "t2"]
    above = bool(tri and cir and tri[0]["bbox"][3] < cir[0]["bbox"][1])
    return {"A1": any(o["fill"] for o in objs), "A6": False, "B1": above, "B3": False, "B4": False,
            "C1": len(objs) == 3}


def _describe(objs, rng):
    order = list(range(len(objs)))
    rng.shuffle(order)
    txt, syms = [], []
    for k, i in enumerate(order, start=1):
        o = objs[i]
        txt.append(f"Shape {k}: {'a filled' if o['fill'] else 'an outline'} {o['shape']}, centre at x={o['x']:.2f} "
                   f"from the left, y={o['y']:.2f} from the top, size {o['radius']:.2f}, rotated {o['rotation']:.0f} degrees.")
        syms.append({"id": k, "shape": o["shape"], "fill": "filled" if o["fill"] else "outline", "x": o["x"],
                     "y": o["y"], "size": o["radius"], "rotation": o["rotation"]})
    rel = []
    for a in range(len(order)):
        for b in range(len(order)):
            if a != b:
                oa, ob = objs[order[a]], objs[order[b]]
                if oa["bbox"][3] < ob["bbox"][1]:
                    rel.append(["above", a + 1, b + 1])
                if oa["bbox"][2] < ob["bbox"][0]:
                    rel.append(["left_of", a + 1, b + 1])
    return " ".join(txt), {"objects": syms, "relations": rel}


def _base_scene(rng, rule, positive):
    # Returns object list satisfying (positive) or violating the rule.
    if rule == "C1":
        n = 3 if positive else 4
        xs = [0.2, 0.5, 0.8, 0.5]
        ys = [0.25, 0.75, 0.25, 0.4]
        return [_obj(i, rng.choice(list(SHAPES)), xs[i], ys[i], 0.09, rng.random() < 0.5, rng) for i in range(n)]
    tx, cx = rng.uniform(0.25, 0.4), rng.uniform(0.6, 0.75)
    ty, cy = (0.25, 0.7) if (positive or rule != "B1") else (0.7, 0.25)
    if rule == "B1":
        fill = rng.random() < 0.5
        return [_obj(0, "triangle", tx, ty, 0.1, fill, rng, "t1"), _obj(1, "circle", cx, cy, 0.1, fill, rng, "t2")]
    if rng.random() < 0.5:
        ty, cy = cy, ty
    filled = positive
    return [_obj(0, "triangle", tx, ty, 0.1, filled, rng, "t1"), _obj(1, "circle", cx, cy, 0.1, False, rng, "t2"),
            _obj(2, "square", 0.5, 0.5, 0.07, False, rng)]


def make_mock_set(root, n_per_rule=4, seed=0, splits=("validation", "test_id", "ood_render")):
    # Builds a tiny contract-v1 fixture set (A1, B1 with reused-panel A1p partners, C1); not real stimuli.
    root, rng = Path(root), random.Random(seed)
    for d in ("panels", "masks", "scenes", "problems", "splits", "audit"):
        (root / d).mkdir(parents=True, exist_ok=True)
    split_ids = {s: [] for s in splits}
    counter = [0]

    def panel(rule, setting, objs, style):
        counter[0] += 1
        pid = f"{rule}-s01-{counter[0]:06d}"
        img, mask = render(objs)
        img.save(root / "panels" / f"{pid}.png")
        Image.fromarray(np.asarray(mask, dtype=np.uint8)).save(root / "masks" / f"{pid}.png")
        scene = {"panel_id": pid, "seed": rng.randrange(2 ** 31), "generator_version": "mock-0",
                 "render_style": style, "objects": objs, "truth": _truth(objs)}
        (root / "scenes" / f"{pid}.json").write_text(json.dumps(scene))
        return pid, objs

    for split in splits:
        for rule, typ in (("B1", "B"), ("A1", "A"), ("C1", "C")):
            for k in range(n_per_rule):
                pos_side = rng.choice(["left", "right"])
                pos = [panel(rule, "s01", _base_scene(rng, rule, True), "default") for _ in range(7)]
                neg = [panel(rule, "s01", _base_scene(rng, rule, False), "default") for _ in range(7)]
                prob_id = f"{rule}-{split}-{k:04d}"
                partner = f"A1p-{split}-{k:04d}" if rule == "B1" else None
                sides = {pos_side: [p for p, _ in pos[:6]], ("right" if pos_side == "left" else "left"): [p for p, _ in neg[:6]]}
                test = [{"panel_id": pos[6][0], "answer": pos_side},
                        {"panel_id": neg[6][0], "answer": "right" if pos_side == "left" else "left"}]
                if rng.random() < 0.5:
                    test.reverse()
                desc, sym = {}, {}
                for pid, objs in pos + neg:
                    desc[pid], sym[pid] = _describe(objs, rng)
                prob = {"problem_id": prob_id, "rule_id": rule, "type": typ, "partner_problem_id": partner,
                        "rule_setting_id": f"{rule}-s01", "knob": 0.1, "split": split, "positive_side": pos_side,
                        "left": sides["left"], "right": sides["right"], "test": test,
                        "twin_map": {p: n for (p, _), (n, _) in zip(pos, neg)}, "described": desc, "symbolic": sym}
                (root / "problems" / f"{prob_id}.json").write_text(json.dumps(prob))
                split_ids[split].append(prob_id)
                if partner:
                    pa = dict(prob, problem_id=partner, rule_id="A1", type="A", partner_problem_id=prob_id,
                              rule_setting_id="A1p-s01")
                    (root / "problems" / f"{partner}.json").write_text(json.dumps(pa))
                    split_ids[split].append(partner)
    for split, ids in split_ids.items():
        (root / "splits" / f"{split}.txt").write_text("\n".join(ids) + "\n")
    manifest = {"contract_version": 1, "generator_version": "mock-0", "git_commit": None,
                "rules": ["A1", "B1", "C1"], "splits": {s: len(v) for s, v in split_ids.items()}}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (root / "audit" / "report.md").write_text("Mock fixture set: no audit.\n")
    return root
