import numpy as np
from scipy import ndimage
from ..generator.render import downsample
from ..generator.predicates import touch_pairs, nest_pairs, ink_gap
from ..generator.scene import poly, stroke_w

INK_THR = 160


def check_panel(img512, sc, size):
    # 5d at one model input size: border clear, expected ink components, stroke and gaps >= 2 px
    im = downsample(img512, size)
    fg = im < INK_THR
    border = fg[:2].any() or fg[-2:].any() or fg[:, :2].any() or fg[:, -2:].any()
    lab, n = ndimage.label(fg, structure=np.ones((3, 3)))
    if n:
        sizes = ndimage.sum(fg, lab, range(1, n + 1))
        n = int((sizes >= 2).sum())
    tp = touch_pairs(sc)
    expect = len(sc["objects"]) - len(tp)
    ob = sc["objects"]
    touching = {frozenset(p) for p in tp}
    nested = {frozenset(p) for p in nest_pairs(sc)}
    gaps = [ink_gap(sc, i, j) for i in range(len(ob)) for j in range(i + 1, len(ob))
            if frozenset((i, j)) not in touching and frozenset((i, j)) not in nested]
    for i, j in nest_pairs(sc):
        gaps.append(poly(ob[i]).exterior.distance(poly(ob[j])) - stroke_w(ob[i]))
    sc.pop("_polys", None)
    min_gap_px = min(gaps) * size if gaps else float("inf")
    stroke_px = min(o["stroke_px"] for o in ob) * size / 512
    return {"no_clip": not border, "components_ok": n == expect, "gap_ok": min_gap_px >= 2.0,
            "stroke_ok": stroke_px >= 1.0, "min_gap_px": float(min_gap_px)}
