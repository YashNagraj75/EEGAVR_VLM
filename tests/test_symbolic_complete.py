import json
import math
import re

import numpy as np
import pytest
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

from eegavr.audit.run import make_pair
from eegavr.eval.dataset import EvalSet
from eegavr.eval.geometry import DECIMALS, OBJECT_KEYS, PANEL_KEYS, complete_object, complete_scene
from eegavr.eval.prompts import PRESENT, build_prompt
from eegavr.eval.vision import SPECS
from eegavr.generator.export import _order, scene_json, symbolic
from eegavr.generator.render import SS, render, render_mask
from eegavr.generator.scene import SHAPES, make_obj, poly
from eegavr.generator.settings import all_settings

RELATION_WORDS = ("role", "inside", "outside", "touch", "touching", "contain", "contains", "enclose", "nested", "above",
                  "below", "under", "over", "left_of", "right_of", "between", "collinear", "same_shape", "relation",
                  "relations", "adjacent", "overlap", "near", "t1", "t2", "distractor", "idx", "seed", "id")
RELATION_RE = re.compile(r"\b(" + "|".join(RELATION_WORDS) + r")\b", re.IGNORECASE)


def _outline(e):
    # Circles are drawn by the generator as regular 96-gons with circumradius r, starting at 90 degrees.
    if "vertices" in e:
        return Polygon(e["vertices"])
    (cx, cy), r = e["centre"], e["radius"]
    a = np.deg2rad(90.0 + 360.0 * np.arange(96) / 96)
    return Polygon(np.c_[cx + r * np.cos(a), cy - r * np.sin(a)])


def _ink(e, width):
    p = _outline(e)
    g = p if e["fill"] == "filled" else p.exterior
    return g.buffer(width / 2, join_style="mitre", mitre_limit=3.0)


def _raster(g, size=512, ss=SS):
    S = size * ss
    im = Image.new("L", (S, S), 0)
    dr = ImageDraw.Draw(im)
    for p in getattr(g, "geoms", [g]):
        dr.polygon([(x * S, y * S) for x, y in p.exterior.coords], fill=255)
        for r in p.interiors:
            dr.polygon([(x * S, y * S) for x, y in r.coords], fill=0)
    return np.asarray(im.reduce(ss)) >= 128


def _iou(a, b):
    return (a & b).sum() / (a | b).sum()


def _objects():
    rng = np.random.default_rng(7)
    out = []
    for shape in SHAPES:
        for fill in (True, False):
            for size in (0.06, 0.11, 0.16):
                for stroke in (3.0, 4.0, 7.5):
                    o = make_obj(shape, size, fill, rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7), rng.uniform(0, 360),
                                 stroke, mirror=bool(rng.random() < 0.5))
                    o["idx"] = 0
                    out.append(o)
    return out


@pytest.mark.parametrize("shape", SHAPES)
def test_full_precision_geometry_reproduces_rendered_ink(shape):
    for o in [o for o in _objects() if o["shape"] == shape]:
        sc = {"objects": [o], "panel_id": "x"}
        e = complete_scene(sc, decimals=None)
        ink = render(sc) < 128
        assert _iou(_raster(_ink(e["objects"][0], e["line_width"])), ink) > 0.98
        assert _iou(_raster(_outline(e["objects"][0]), ss=1), render_mask(sc) > 0) > 0.98


def test_prompt_precision_vertices_reproduce_filled_region():
    for o in _objects():
        e, exact = complete_object(o), poly(o)
        rec = _outline(e)
        assert rec.intersection(exact).area / rec.union(exact).area > 0.98
        if "vertices" in e:
            err = np.abs(np.array(e["vertices"]) - np.array(exact.exterior.coords)[:-1]).max()
            assert err <= 0.5 * 10 ** -DECIMALS + 1e-12


def test_order_matches_exported_symbolic_list_and_hides_ids():
    st = all_settings()["B3-s03"]
    for i in range(6):
        sc = make_pair(st, i)["pos"]
        pid = f"B3-s03-{i:010d}"
        sj = scene_json(sc, pid, 1234 + i, {})
        sym = symbolic(sc, _order(pid, len(sc["objects"])))["objects"]
        comp = complete_scene(sj)
        assert set(comp) == set(PANEL_KEYS) and len(comp["objects"]) == len(sym)
        for e, s in zip(comp["objects"], sym):
            assert set(e) <= set(OBJECT_KEYS) and (e["shape"], e["fill"]) == (s["shape"], s["fill"])
            c = e["centre"] if "centre" in e else list(Polygon(e["vertices"]).centroid.coords[0])
            assert math.hypot(c[0] - s["x"], c[1] - s["y"]) < 2e-3
        assert not RELATION_RE.search(json.dumps(comp)) and pid not in json.dumps(comp)


def test_mixed_stroke_widths_rejected():
    a = make_obj("square", 0.1, True, 0.3, 0.3, 0.0, 4.0)
    b = make_obj("circle", 0.1, True, 0.7, 0.7, 0.0, 4.5)
    with pytest.raises(ValueError):
        complete_scene({"panel_id": "x", "objects": [a, b]})


def test_symbolic_complete_prompt_is_neutral(mockset):
    es = EvalSet(mockset)
    assert not RELATION_RE.search(PRESENT[("symbolic_complete", None)])
    for split in es.splits():
        for pid in es.split_ids(split):
            prob = es.load_problem(pid)
            text = build_prompt(prob, "symbolic_complete", "none", SPECS["qwen3vl"], es.load_panel,
                                es.load_scene).text_with_placeholders()
            lines = [ln for ln in text.splitlines() if re.match(r"^[LRT]\d: ", ln)]
            assert len(lines) == 14
            for ln in lines:
                payload = json.loads(ln.split(": ", 1)[1])
                assert set(payload) == set(PANEL_KEYS) and all(set(o) <= set(OBJECT_KEYS) for o in payload["objects"])
                assert not RELATION_RE.search(ln[4:])
    with pytest.raises(ValueError):
        build_prompt(prob, "symbolic_complete", "none", SPECS["qwen3vl"], es.load_panel)
