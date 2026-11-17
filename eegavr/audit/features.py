import numpy as np
from PIL import Image
from scipy import ndimage
from ..generator import rules_borient as _borient

T1_NAMES = ["mean_brightness", "brightness_var", "foreground_frac", "ink_count", "r_mean", "g_mean", "b_mean",
            "saturation", "edge_density", "outline_length", "connected_components", "dir_horizontal",
            "dir_vertical", "dir_diagonal", "estimated_count", "occupied_area", "rect_w", "rect_h", "rect_aspect",
            "rect_area", "centre_x", "centre_y", "inter_region_dist", "pixel_spread", "object_spread"]
POS_NAMES = ["ink_cx", "ink_cy", "ink_median_x", "ink_median_y", "ink_top_y", "ink_bottom_y", "cx_in_bbox",
             "cy_in_bbox", "quad_tl", "quad_tr", "quad_bl", "quad_br"]
ROLE_NAMES = [f"{r}_{k}" for r in ("t1", "t2") for k in ("ink", "cx_in_own_bbox", "cy_in_own_bbox", "own_h", "own_w")]
ROLE_ABS_NAMES = [f"{r}_{k}" for r in ("t1", "t2") for k in ("abs_cx", "abs_cy")]
REGION_NAMES = ["connected_components", "estimated_count", "inter_region_dist", "object_spread"]
ALL_NAMES = T1_NAMES + POS_NAMES + ROLE_NAMES + ROLE_ABS_NAMES


def track1(img512):
    # the 25 Track 1 measurements at 128 x 128 (grayscale: R=G=B=brightness, saturation=0)
    g = np.asarray(Image.fromarray(img512).reduce(4), float) / 255.0
    H, W = g.shape
    ink = 1.0 - g
    fg = g < 0.5
    gy, gx = np.gradient(g)
    mag = np.hypot(gx, gy)
    e = mag > 0.1
    ang = np.mod(np.arctan2(gy, gx), np.pi)
    w = mag[e]
    a = ang[e]
    hor = np.abs(np.sin(a)) ** 2
    ver = np.abs(np.cos(a)) ** 2
    dsum = w.sum() + 1e-9
    d_h, d_v = (w * hor).sum() / dsum, (w * ver).sum() / dsum
    d_d = (w * np.abs(np.sin(2 * a))).sum() / dsum
    outline = fg & ~ndimage.binary_erosion(fg)
    lab, ncomp = ndimage.label(fg, structure=np.ones((3, 3)))
    sizes = ndimage.sum(fg, lab, range(1, ncomp + 1)) if ncomp else np.array([])
    keep = [i + 1 for i, s in enumerate(sizes) if s >= max(3, 0.0005 * H * W)]
    ys, xs = np.nonzero(fg)
    if len(xs) == 0:
        return [0.0] * 25
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    rw, rh = (x1 - x0 + 1) / W, (y1 - y0 + 1) / H
    cents = np.array(ndimage.center_of_mass(fg, lab, keep)) / H if keep else np.zeros((1, 2))
    if len(cents) > 1:
        dd = np.hypot(*(cents[:, None, :] - cents[None, :, :]).transpose(2, 0, 1))
        ird = dd[np.triu_indices(len(cents), 1)].mean()
        osp = cents.std(axis=0).mean()
    else:
        ird, osp = 0.0, 0.0
    m = g.mean()
    return [m, g.var(), fg.mean(), ink.sum() / (H * W), m, m, m, 0.0, e.mean(), outline.sum() / (H * W),
            float(ncomp), d_h, d_v, d_d, float(len(keep)), np.isin(lab, keep).mean(), rw, rh, rw / rh, rw * rh,
            (x0 + x1) / 2 / W, (y0 + y1) / 2 / H, ird, np.sqrt(xs.var() + ys.var()) / W, osp]


def ink_position(img512):
    # at 128 px like Track 1: weighted centroid, medians, top/bottom ink height, centroid in bbox, quadrant ink
    ink = 1.0 - np.asarray(Image.fromarray(img512).reduce(4), float) / 255.0
    H, W = ink.shape
    ys, xs = np.nonzero(ink > 0.5)
    tot = ink.sum() + 1e-9
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = (ink * xx).sum() / tot / W, (ink * yy).sum() / tot / H
    x0, x1, y0, y1 = xs.min() / W, xs.max() / W, ys.min() / H, ys.max() / H
    h, w = H // 2, W // 2
    q = [ink[:h, :w].sum() / tot, ink[:h, w:].sum() / tot, ink[h:, :w].sum() / tot, ink[h:, w:].sum() / tot]
    return [cx, cy, np.median(xs) / W, np.median(ys) / H, y0, y1, (cx - x0) / max(x1 - x0, 1e-6),
            (cy - y0) / max(y1 - y0, 1e-6)] + q


def role_features(img512, mask, scene):
    # per role, from that role rendered alone: ink mass, centroid within its own ink bbox, extent; plus absolute centroid
    from ..generator.render import render
    rel, ab = [], []
    for r in ("t1", "t2"):
        o = [o for o in scene["objects"] if o["role"] == r]
        if not o:
            rel += [0.0] * 5
            ab += [0.0] * 2
            continue
        ink = 1.0 - render({"objects": o[:1], "render_style": scene.get("render_style", "default")}).astype(float) / 255
        H, W = ink.shape
        ys, xs = np.nonzero(ink > 0.5)
        t = ink.sum()
        yy, xx = np.mgrid[0:H, 0:W]
        cx, cy = (ink * xx).sum() / t / W, (ink * yy).sum() / t / H
        x0, x1, y0, y1 = xs.min() / W, xs.max() / W, ys.min() / H, ys.max() / H
        rel += [t / (H * W), (cx - x0) / max(x1 - x0, 1e-6), (cy - y0) / max(y1 - y0, 1e-6), y1 - y0, x1 - x0]
        ab += [cx, cy]
    return rel + ab


def pixel_features(img512, mask, scene):
    return np.array(track1(img512) + ink_position(img512) + role_features(img512, mask, scene), float)


def pass_columns(rule):
    # pass probe (lead, 2026-11-16): role-free global features minus the rule's relation-constitutive ones
    drop = _borient.PIX_RELATION.get(rule, ())
    return [i for i in global_columns() if ALL_NAMES[i] not in drop]


def global_columns():
    return [i for i, n in enumerate(ALL_NAMES) if n in T1_NAMES + POS_NAMES and n not in REGION_NAMES]
