import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union
from .scene import PANEL_PX, ink_geom, poly, stroke_w

SS = 4
HATCH_SPACING = 0.025
STYLES = ("default", "hatch")


def _polys(g):
    return list(g.geoms) if hasattr(g, "geoms") else [g]


def _hatch(o):
    # OOD style: filled region drawn as 45-degree hatching inside a solid 1.5x boundary
    p, w = poly(o), stroke_w(o)
    x0, y0, x1, y1 = p.bounds
    d = (x1 - x0) + (y1 - y0)
    lines = [LineString([(x0 + t, y0), (x0 + t - d, y0 + d)]) for t in np.arange(0, 2 * d, HATCH_SPACING)]
    h = MultiLineString(lines).intersection(p).buffer(w / 2, cap_style="flat")
    return unary_union([h, p.exterior.buffer(w / 2, join_style="mitre", mitre_limit=3.0)])


def object_ink(o, style="default"):
    if style == "hatch" and o["fill"]:
        return _hatch(o)
    return ink_geom(o, style)


def render(scene, size=PANEL_PX, style=None):
    # 8-bit grayscale, black ink on white, antialiased by 4x supersampling
    style = style or scene.get("render_style", "default")
    S = size * SS
    canvas = Image.new("L", (S, S), 0)
    for o in scene["objects"]:
        g = object_ink(o, style)
        layer = Image.new("L", (S, S), 0)
        dr = ImageDraw.Draw(layer)
        for p in _polys(g):
            if p.is_empty:
                continue
            dr.polygon([(x * S, y * S) for x, y in p.exterior.coords], fill=255)
            for r in p.interiors:
                dr.polygon([(x * S, y * S) for x, y in r.coords], fill=0)
        canvas = Image.fromarray(np.maximum(np.asarray(canvas), np.asarray(layer)))
    small = canvas.reduce(SS)
    return 255 - np.asarray(small, dtype=np.uint8)


def render_mask(scene, size=PANEL_PX):
    # label map: 0 background, k+1 = object k's filled region; larger objects drawn first
    m = Image.new("L", (size, size), 0)
    dr = ImageDraw.Draw(m)
    objs = sorted(scene["objects"], key=lambda o: -poly(o).area)
    for o in objs:
        dr.polygon([(x * size, y * size) for x, y in poly(o).exterior.coords], fill=o["idx"] + 1)
    return np.asarray(m, dtype=np.uint8)


def downsample(img, size, method="box"):
    # area averaging (PIL BOX) as in the eval harness; bicubic kept for the logged 0.2.1 setting audits
    f = Image.BOX if method == "box" else Image.BICUBIC
    return np.asarray(Image.fromarray(img).resize((size, size), f), dtype=np.uint8)
