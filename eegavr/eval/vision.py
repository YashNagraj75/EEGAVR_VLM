import math
from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFont

RESAMPLE = Image.Resampling.BOX  # area averaging; the generator's 5d check should use the same


@dataclass(frozen=True)
class VisionSpec:
    # Per-model image geometry. The harness resizes/pads itself so the model processor's own
    # resize is a no-op (Qwen: dims are multiples of 32; InternVL: exact 448-px tile grids).
    family: str
    separate_panel_px: int
    composite_panel_px: int
    qwen_factor: int = 32
    qwen_min_pixels: int = 65536
    qwen_max_pixels: int = 16777216
    tile_px: int = 448
    tokens_per_tile: int = 256
    max_tiles: int = 12
    use_thumbnail: bool = True
    composite_grid: tuple = field(default=(4, 3))


SPECS = {
    "qwen3vl": VisionSpec("qwen", separate_panel_px=512, composite_panel_px=384),
    "internvl": VisionSpec("internvl", separate_panel_px=448, composite_panel_px=256, composite_grid=(4, 3)),
}
SPECS["mock"] = SPECS["qwen3vl"]


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except (TypeError, OSError):
        return ImageFont.load_default()


def composite_geometry(p):
    # Layout in pixels for panel size p. Test panels are centred, not under either block.
    m, g, b, hdr, gap, tgap = (round(p * f) for f in (0.06, 0.06, 0.14, 0.16, 0.5, 0.25))
    block_w = 3 * p + 2 * g
    W = 2 * m + 2 * block_w + gap
    H = 2 * m + hdr + 2 * (b + p) + g + tgap + hdr + b + p
    boxes = {}
    y_rows = [m + hdr + b, m + hdr + b + p + g + b]
    for side, x0 in (("L", m), ("R", m + block_w + gap)):
        for r in range(2):
            for c in range(3):
                boxes[f"{side}{1 + 3 * r + c}"] = (x0 + c * (p + g), y_rows[r])
    y_test = y_rows[1] + p + tgap + hdr + b
    tx0 = (W - (2 * p + gap)) // 2
    boxes["T1"] = (tx0, y_test)
    boxes["T2"] = (tx0 + p + gap, y_test)
    headers = {"LEFT": (m + block_w // 2, m), "RIGHT": (m + block_w + gap + block_w // 2, m),
               "TEST": (W // 2, y_rows[1] + p + tgap)}
    return dict(W=W, H=H, p=p, label_band=b, header=hdr, boxes=boxes, headers=headers)


def composite_canvas(spec, W, H):
    # Target canvas (w, h) for the padded composite.
    if spec.family == "qwen":
        f = spec.qwen_factor
        return math.ceil(W / f) * f, math.ceil(H / f) * f
    cols, rows = spec.composite_grid
    cw, ch = cols * spec.tile_px, rows * spec.tile_px
    if W > cw or H > ch:
        raise ValueError(f"composite {W}x{H} does not fit tile grid {cw}x{ch}")
    return cw, ch


def render_composite(panels, spec):
    # panels: dict label -> PIL image (any size, resized here to the composite panel size).
    p = spec.composite_panel_px
    geo = composite_geometry(p)
    cw, ch = composite_canvas(spec, geo["W"], geo["H"])
    ox, oy = (cw - geo["W"]) // 2, (ch - geo["H"]) // 2
    canvas = Image.new("RGB", (cw, ch), "white")
    draw = ImageDraw.Draw(canvas)
    lab_font, hdr_font = _font(max(10, round(0.11 * p))), _font(max(12, round(0.13 * p)))
    for name, (cx, y) in geo["headers"].items():
        draw.text((ox + cx, oy + y + geo["header"] // 2), name, fill="black", font=hdr_font, anchor="mm")
    for label, (x, y) in geo["boxes"].items():
        im = panels[label]
        if im.size != (p, p):
            im = im.resize((p, p), RESAMPLE)
        canvas.paste(im.convert("RGB"), (ox + x, oy + y))
        draw.rectangle([ox + x - 2, oy + y - 2, ox + x + p + 1, oy + y + p + 1], outline=(128, 128, 128), width=2)
        draw.text((ox + x + p // 2, oy + y - geo["label_band"] // 2), label, fill="black", font=lab_font, anchor="mm")
    return canvas


def prepare_separate(im, spec):
    p = spec.separate_panel_px
    im = im if im.size == (p, p) else im.resize((p, p), RESAMPLE)
    return im.convert("RGB")


def qwen_smart_resize(h, w, factor=32, min_pixels=65536, max_pixels=16777216):
    # Copy of transformers qwen2_vl smart_resize (5.23), used to predict the processor's resize.
    h_bar, w_bar = round(h / factor) * factor, round(w / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar, w_bar = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
    return h_bar, w_bar


def internvl_grid(w, h, tile=448, max_tiles=12, use_thumbnail=True, vllm_rule=True):
    # Closest-aspect tile grid as in vLLM 0.33 (max+1 when thumbnail) and HF GotOcr2 (max).
    max_num = max_tiles + 1 if (vllm_rule and use_thumbnail and max_tiles != 1) else max_tiles
    ratios = sorted({(i, j) for n in range(1, max_num + 1) for i in range(1, n + 1)
                     for j in range(1, n + 1) if 1 <= i * j <= max_num}, key=lambda r: r[0] * r[1])
    best, best_diff, ar = (1, 1), float("inf"), w / h
    for r in ratios:
        d = abs(ar - r[0] / r[1])
        if d < best_diff:
            best, best_diff = r, d
        elif d == best_diff and w * h > 0.5 * tile * tile * r[0] * r[1]:
            best = r
    return best


def image_cost(spec, w, h):
    # (visual tokens incl. wrapper tokens, scale factor applied by the model processor).
    if spec.family == "qwen":
        rh, rw = qwen_smart_resize(h, w, spec.qwen_factor, spec.qwen_min_pixels, spec.qwen_max_pixels)
        return (rh // spec.qwen_factor) * (rw // spec.qwen_factor) + 2, rw / w
    cols, rows = internvl_grid(w, h, spec.tile_px, spec.max_tiles, spec.use_thumbnail)
    tiles = cols * rows + (1 if spec.use_thumbnail and cols * rows > 1 else 0)
    return tiles * spec.tokens_per_tile + 2, cols * spec.tile_px / w


def layout_summary(spec):
    # Effective panel size (px seen by the vision encoder) and visual tokens per layout.
    p = spec.separate_panel_px
    tok_sep, s_sep = image_cost(spec, p, p)
    geo = composite_geometry(spec.composite_panel_px)
    cw, ch = composite_canvas(spec, geo["W"], geo["H"])
    tok_comp, s_comp = image_cost(spec, cw, ch)
    return {
        "separate": {"panel_px": p, "effective_panel_px": round(p * s_sep, 1), "images": 14,
                     "visual_tokens": 14 * tok_sep},
        "composite": {"panel_px": spec.composite_panel_px, "canvas": [cw, ch],
                      "effective_panel_px": round(spec.composite_panel_px * s_comp, 1), "images": 1,
                      "visual_tokens": tok_comp},
    }
