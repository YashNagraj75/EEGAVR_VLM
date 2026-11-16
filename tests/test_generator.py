import numpy as np
import pytest
from shapely.geometry import Polygon
from eegavr.generator.scene import make_obj, index, poly, UNIT, SHAPES, CHIRAL, ink_area
from eegavr.generator import predicates as P
from eegavr.generator.rules import sample_pair, check_twin
from eegavr.generator.settings import all_settings, id_settings, seed_for
from eegavr.generator.render import render, render_mask

ST = all_settings()


def sc(*objs):
    return {"objects": index(list(objs))}


def test_perimeters_normalised():
    for k in SHAPES:
        assert abs(Polygon(UNIT[k]).length - 2 * np.pi) < 1e-9


def test_position_is_area_centroid():
    o = make_obj("L", 0.1, True, 0.4, 0.6, 33.0, 4.0)
    c = poly(o).centroid
    assert abs(c.x - 0.4) < 1e-9 and abs(c.y - 0.6) < 1e-9


def test_chiral_shapes_have_no_mirror_symmetry():
    for k in SHAPES:
        if not CHIRAL[k]:
            continue
        A = poly(make_obj(k, 0.1, False, .5, .5, 0, 4))
        best = min(A.symmetric_difference(poly(make_obj(k, 0.1, False, .5, .5, r, 4, mirror=True))).area
                   for r in np.arange(0, 360, 0.5))
        assert best > 0.05 * A.area


def test_a1_a6():
    ft = make_obj("triangle", .1, True, .3, .3, 0, 4)
    os_ = make_obj("square", .1, False, .7, .7, 0, 4)
    assert P.a1(sc(ft, os_)) and P.a6(sc(ft, os_))
    assert P.a1(sc(dict(ft, fill=False), dict(os_, fill=True)))
    assert not P.a6(sc(dict(ft, fill=False), dict(os_, fill=True)))
    assert not P.a1(sc(dict(ft, fill=False), os_))


def test_b1_extent_based():
    c = make_obj("circle", .08, False, .5, .60, 0, 4)
    assert P.b1(sc(make_obj("triangle", .08, False, .5, .25, 0, 4), c))
    assert not P.b1(sc(make_obj("triangle", .08, False, .5, .50, 0, 4), c))
    assert not P.b1(sc(make_obj("triangle", .08, False, .5, .85, 0, 4), c))


def test_b3_inside_and_margin():
    o = make_obj("circle", .15, False, .5, .5, 0, 4)
    i = make_obj("square", .06, False, .5, .5, 0, 4)
    s = sc(o, i)
    assert P.b3(s) and P.b3(s, margin=0.03) and not P.b3(s, margin=0.12)
    assert not P.b4(s)
    assert not P.b3(sc(o, make_obj("square", .06, False, .85, .85, 0, 4)))


def test_b4_touching_vs_gap():
    a = make_obj("square", .1, False, .3, .5, 0, 4)
    half = a["bbox"][2] - .3
    b = make_obj("square", .1, False, .3 + 2 * half + 1e-7, .5, 0, 4)
    assert P.b4(sc(a, b))
    c = make_obj("square", .1, False, .3 + 2 * half + 0.03, .5, 0, 4)
    assert not P.b4(sc(a, c))
    big = make_obj("circle", .15, False, .5, .5, 0, 4)
    assert not P.b4(sc(big, make_obj("circle", .05, False, .5, .5, 0, 4)))


def test_c1_count():
    objs = [make_obj("circle", .06, False, .2 + .2 * k, .5, 0, 4) for k in range(4)]
    assert P.c1(sc(*objs[:3])) and not P.c1(sc(*objs))


@pytest.mark.parametrize("sid", [s["id"] for r in ("A1", "A6", "B1", "B3", "B4", "C1") for s in id_settings(r)])
def test_pairs_valid_and_deterministic(sid):
    st = ST[sid]
    for i in range(3):
        ctrl = ("equal_ink" if i % 2 else "equal_size") if st["rule"] == "C1" else None
        a = sample_pair(st, seed_for("audit", sid, i), control=ctrl)
        b = sample_pair(st, seed_for("audit", sid, i), control=ctrl)
        assert a["pos"]["objects"] == b["pos"]["objects"] and a["twin"]["objects"] == b["twin"]["objects"]
        assert a["pos"]["truth"][st["rule"]] and not a["twin"]["truth"][st["rule"]]
        assert np.array_equal(render(a["pos"]), render(b["pos"]))
        assert np.array_equal(render_mask(a["twin"]), render_mask(b["twin"]))


def test_checker_rejects_undeclared_change():
    st = ST["B1-s02"]
    p = sample_pair(st, seed_for("audit", "B1-s02", 0))
    twin = {"objects": [dict(o) for o in p["twin"]["objects"]]}
    twin["objects"][0]["x"] += 0.01
    v = check_twin("B1", p["pos"], twin, {(0, "y"), (1, "y")})
    assert any("undeclared change obj0.x" in m for m in v)


def test_checker_rejects_twin_that_keeps_rule():
    p = sample_pair(ST["A1-s02"], seed_for("audit", "A1-s02", 0))
    v = check_twin("A1", p["pos"], p["pos"], {(0, "fill")})
    assert "twin still satisfies rule" in v


def test_checker_rejects_ink_change_in_binding_rule():
    p = sample_pair(ST["A6-s02"], seed_for("audit", "A6-s02", 0))
    tw = {"objects": [dict(o) for o in p["twin"]["objects"]]}
    tw["objects"][1] = dict(tw["objects"][1], size=tw["objects"][1]["size"] * 1.3)
    v = check_twin("A6", p["pos"], tw, {(0, "fill"), (1, "fill")})
    assert any(m.startswith("ink changed") for m in v)


def test_seeds_disjoint_across_splits():
    seen = set()
    for split in ("audit", "validation", "test_id", "ood_count", "ood_count_control", "ood_render"):
        s = {seed_for(split, sid, i) for sid in ST for i in (0, 999_999)}
        assert not (s & seen)
        seen |= s
