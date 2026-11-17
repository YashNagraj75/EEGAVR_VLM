import numpy as np
import pytest
from eegavr.generator.scene import make_obj, index, ink_area
from eegavr.generator import rules_a as RA
from eegavr.generator.predicates import RULE_FN, truth, candidate_predicates
from eegavr.generator.rules import sample_pair, check_twin
from eegavr.generator.settings import all_settings, seed_for, SETTING_CODE, id_settings, ood_settings
from eegavr.generator.render import render

ST = all_settings()
A_SIDS = [s["id"] for r in RA.A_RULES for s in id_settings(r) + list(ood_settings(r).values())]
PILOT_CODES = {"A1-s01": 1, "A1-or": 6, "A6-s01": 7, "B1-s01": 13, "B3-or": 24, "B4-or": 30, "C1-or": 33,
               "A1pB1-s01": 34, "A6pB3-s01": 35, "A1pB4-s01": 36, "C1b-s01": 37, "C1b-or": 40}


def sc(*objs):
    return {"objects": index(list(objs))}


def test_pilot_setting_codes_unchanged():
    for sid, code in PILOT_CODES.items():
        assert SETTING_CODE[sid] == code
    assert max(SETTING_CODE[s] for s in A_SIDS) > 40 and len(A_SIDS) == 24


def test_a3_threshold():
    assert RA.a3(sc(make_obj("square", 0.12, False, .3, .3, 0, 4), make_obj("circle", .07, False, .7, .7, 0, 4)))
    assert not RA.a3(sc(make_obj("square", 0.10, False, .3, .3, 0, 4)))


def test_a4_tilt_modulo_symmetry():
    assert RA.tilt(make_obj("square", .1, False, .5, .5, 135.0, 4)) == pytest.approx(45.0)
    assert RA.tilt(make_obj("triangle", .1, False, .5, .5, 240.0, 4)) == pytest.approx(0.0)
    assert RA.tilt(make_obj("pentagon", .1, False, .5, .5, 45.0, 4)) == pytest.approx(27.0)
    assert not RA.a4(sc(make_obj("circle", .1, False, .5, .5, 45.0, 4)))
    assert RA.a4(sc(make_obj("L", .1, False, .5, .5, 30.0, 4)))
    assert RA.a4_near45(sc(make_obj("cross", .1, False, .5, .5, 135.0, 4)))


def test_a5_concave_and_depth():
    c = make_obj("cross", .1, False, .5, .5, 0, 4)
    assert RA.a5(sc(c)) and RA.a5(sc(c), depth=0.03) and not RA.a5(sc(c), depth=0.05)
    assert not RA.a5(sc(make_obj("right_trapezoid", .1, False, .5, .5, 0, 4)))


def test_registries():
    assert all(r in RULE_FN for r in RA.A_RULES)
    t = truth({"objects": [make_obj("cross", .12, False, .5, .5, 30, 4)]})
    assert t["A3"] and t["A4"] and t["A5"] and not t["A2"]
    C = candidate_predicates()
    assert "A4:tilt>=10 present" in C and "A5:concave depth>=0.03 present" in C


@pytest.mark.parametrize("sid", A_SIDS)
def test_a_pairs_valid_and_deterministic(sid):
    st = ST[sid]
    for i in range(3):
        a = sample_pair(st, seed_for("audit", sid, i))
        b = sample_pair(st, seed_for("audit", sid, i))
        assert a["pos"]["objects"] == b["pos"]["objects"] and a["twin"]["objects"] == b["twin"]["objects"]
        assert a["pos"]["truth"][st["rule"]] and not a["twin"]["truth"][st["rule"]]
        P, T = a["pos"]["objects"], a["twin"]["objects"]
        assert abs(RA.min_gap(P) - RA.min_gap(T)) < 1e-9
        assert RA.rule_diff(a) >= 2.0
        if st["rule"] in ("A2", "A4", "A5"):
            ip, it = sum(ink_area(o) for o in P), sum(ink_area(o) for o in T)
            assert abs(ip - it) / ip <= RA.INK_TOL
        if st["rule"] == "A4":
            assert all(RA.tilt(o) < 1e-9 for o in P[1:] + T if o["shape"] != "circle")
    assert np.array_equal(render(a["pos"]), render(b["pos"]))


def test_a2_checker_rejects_pentagon_kept_as_triangle():
    p = sample_pair(ST["A2-s02"], seed_for("audit", "A2-s02", 0))
    v = check_twin("A2", p["pos"], p["pos"], {(0, "shape")})
    assert "twin still satisfies rule" in v
