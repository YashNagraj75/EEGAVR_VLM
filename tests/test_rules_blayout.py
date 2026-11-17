import numpy as np
import pytest
from eegavr.generator.scene import make_obj, index
from eegavr.generator import rules_blayout as BL
from eegavr.generator.predicates import RULE_FN, truth, candidate_predicates
from eegavr.generator.rules import sample_pair
from eegavr.generator.settings import all_settings, id_settings, ood_settings, seed_for, SETTING_CODE
from eegavr.audit.competing import FAMILY

ST = all_settings()


def sc(*objs):
    return {"objects": index(list(objs))}


def o(shape, x, y, size=0.06):
    return make_obj(shape, size, False, x, y, 0, 4)


def test_registered():
    for r in BL.RULES:
        assert r in RULE_FN and r in FAMILY and r in truth(sc(o("circle", .5, .5)))
        assert all(st["rule"] == r for st in id_settings(r) + list(ood_settings(r).values()))
    names = list(candidate_predicates())
    assert all(any(n.startswith(p) for n in names) for f in BL.FAMILY.values() for p in f)


def test_new_settings_coded_after_pilot():
    pilot = [k for k in ST if k.split("-")[0] not in BL.RULES]
    new = [k for k in ST if k.split("-")[0] in BL.RULES]
    assert max(SETTING_CODE[k] for k in pilot) < min(SETTING_CODE[k] for k in new)


def test_b2_left_of_extent():
    c = o("circle", .6, .5)
    assert BL.left_of(sc(o("triangle", .25, .5), c))
    assert not BL.left_of(sc(o("triangle", .55, .5), c))
    assert not BL.left_of(sc(o("triangle", .85, .5), c))


def test_b5_collinear_and_deviation():
    a, b = o("square", .2, .2), o("cross", .8, .8)
    assert BL.b5(sc(a, o("circle", .4, .4), b))
    assert not BL.b5(sc(a, o("circle", .4, .43), b))
    d, k = BL.triple_dev(a, o("circle", .5, .5 + 0.03 * np.sqrt(2)), b)
    assert abs(d - 0.03) < 1e-9 and k == 1


def test_b8_odd_between_same():
    s1, s2 = o("square", .2, .5), o("square", .8, .5)
    assert BL.b8(sc(s1, o("circle", .5, .5), s2))
    assert not BL.b8(sc(o("circle", .2, .5), o("square", .5, .5), s2))
    assert not BL.b8(sc(s1, o("circle", .5, .55), s2))


def test_b9_closest_pair_same_shaped():
    s1, s2 = o("square", .2, .2), o("square", .4, .2)
    assert BL.b9(sc(s1, s2, o("circle", .7, .7)))
    assert not BL.b9(sc(s1, o("circle", .4, .2), o("square", .7, .7)))
    assert not BL.b9(sc(s1, s2, o("circle", .7, .7)), ratio=10.0)


@pytest.mark.parametrize("sid", [k for k in ST if k.split("-")[0] in BL.RULES])
def test_pairs_valid_and_deterministic(sid):
    st = ST[sid]
    for i in range(2):
        a = sample_pair(st, seed_for("audit", sid, i))
        b = sample_pair(st, seed_for("audit", sid, i))
        assert a["pos"]["objects"] == b["pos"]["objects"] and a["twin"]["objects"] == b["twin"]["objects"]
        assert a["pos"]["truth"][st["rule"]] and not a["twin"]["truth"][st["rule"]]
        assert st["knob"][0] - 1e-6 <= a["knob"] <= st["knob"][1] + 1e-6
        pa = sorted((q["x"], q["y"]) for q in a["pos"]["objects"])
        pt = sorted((q["x"], q["y"]) for q in a["twin"]["objects"])
        if st["rule"] in ("B8", "B9"):
            assert np.allclose(pa, pt)


@pytest.mark.parametrize("sid", ["B8-s01", "B8-oc", "B9-s01"])
def test_twin_layout_valid(sid):
    from eegavr.generator.scene import layout_ok
    for i in range(20):
        a = sample_pair(ST[sid], seed_for("audit", sid, i))
        assert layout_ok(a["pos"]["objects"]) and layout_ok(a["twin"]["objects"])


@pytest.mark.parametrize("sid", ["B8-oc", "B9-occ"])
def test_no_confusable_shapes_in_identity_rules(sid):
    for i in range(20):
        sh = {q["shape"] for q in sample_pair(ST[sid], seed_for("audit", sid, i))["pos"]["objects"]}
        assert not {"pentagon", "irregular_pentagon"} <= sh
