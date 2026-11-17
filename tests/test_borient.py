import hashlib
import json
import numpy as np
from shapely import affinity
from shapely.geometry import Polygon
from eegavr.generator.scene import make_obj, index, poly, UNIT, ink_area
from eegavr.generator import rules_borient as R
from eegavr.generator.predicates import RULE_FN, truth, candidate_predicates
from eegavr.generator.rules import sample_pair
from eegavr.generator.settings import all_settings, seed_for, SETTING_CODE

ST = all_settings()
BO = [k for k in ST if ST[k]["rule"] in ("B6", "B7")]


def sc(*objs):
    return {"objects": index(list(objs))}


def test_b6_modulo_symmetry():
    a = make_obj("square", .08, False, .3, .3, 10, 4)
    assert R.b6(sc(a, make_obj("square", .08, False, .7, .7, 100, 4)))
    assert not R.b6(sc(a, make_obj("square", .08, False, .7, .7, 40, 4)))
    t = make_obj("triangle", .08, False, .3, .3, 5, 4)
    assert R.b6(sc(t, make_obj("triangle", .08, False, .7, .7, 245, 4)))
    l1 = make_obj("L", .08, False, .3, .3, 30, 4, mirror=True)
    assert not R.b6(sc(l1, make_obj("L", .08, False, .7, .7, 30, 4, mirror=False)))
    assert not R.b6(sc(make_obj("circle", .08, False, .3, .3, 0, 4), make_obj("circle", .08, False, .7, .7, 0, 4)))


def test_b7_vertical_reflection_geometry():
    for sh in R.CHIRAL_SHAPES:
        a = make_obj(sh, .1, False, .5, .5, 37, 4)
        b = make_obj(sh, .1, False, .5, .5, -37, 4, mirror=True)
        ref = affinity.scale(poly(a), -1, 1, origin=(.5, .5))
        assert ref.symmetric_difference(poly(b)).area < 1e-9
        assert R.b7(sc(a, dict(b, x=.8))) and not R.b7(sc(a, dict(a, x=.8, rotation=80.0)))


def test_chirality_constants_are_lower_bounds():
    for sh, h in R.CHIRAL_HAUSDORFF.items():
        P, M = Polygon(UNIT[sh]), Polygon(UNIT[sh] * [-1, 1])
        best = min(M.hausdorff_distance(affinity.rotate(P, r, origin=(0, 0))) for r in np.arange(0, 360, 1.0))
        assert best >= h


def test_registered_without_moving_pilot_codes():
    assert {"B6", "B7"} <= set(RULE_FN) and {"B6", "B7"} <= set(truth({"objects": []}))
    assert SETTING_CODE["A1-s01"] == 1 and SETTING_CODE["B4-or"] == 30 and SETTING_CODE["A1pB4-s01"] == 36
    assert SETTING_CODE["C1b-or"] == 40 and min(SETTING_CODE[k] for k in BO) == 41
    assert any(k.startswith("B6:") for k in candidate_predicates()) and "B7:mirror pair" in candidate_predicates()


def _key(objs):
    return [[o["shape"], o["fill"], bool(o["mirror"]), o["role"]] +
            [round(float(o[k]), 7) for k in ("x", "y", "size", "rotation", "stroke_px")] for o in objs]


def test_pilot_pairs_unchanged():
    ref = {"A1-s01": "486e4d94d3fc", "A6-s02": "d4bea10dc1f7", "B1-s02": "fa6889e26e03", "B3-s03": "51440c2a9e58",
           "B4-s01": "9649bbf11c74", "C1b-s01": "576349ab3114", "B4-or": "569dbf9d4450"}
    for sid, h in ref.items():
        p = sample_pair(ST[sid], seed_for("test_id", sid, 3))
        s = json.dumps([_key(p["pos"]["objects"]), _key(p["twin"]["objects"])])
        assert hashlib.sha1(s.encode()).hexdigest()[:12] == h, sid


def test_borient_pairs_valid():
    for sid in BO:
        st = ST[sid]
        for i in range(8):
            p = sample_pair(st, seed_for("audit", sid, i))
            P, T = p["pos"]["objects"], p["twin"]["objects"]
            assert p["pos"]["truth"][st["rule"]] and not p["twin"]["truth"][st["rule"]]
            assert st["knob"][0] - 1e-6 <= p["knob"] <= st["knob"][1] + 1e-6
            assert abs(sum(map(ink_area, P)) - sum(map(ink_area, T))) < 1e-4 * sum(map(ink_area, P))
            assert P[0]["shape"] == P[1]["shape"] and P[0]["size"] == P[1]["size"] and P[0]["fill"] == P[1]["fill"]
            assert P[1]["shape"] not in [o["shape"] for o in P[2:]]
            if st["rule"] == "B6":
                assert R.rotation_diff_px(T[1], R.angle_diff(T[0], T[1])) >= R.MIN_DIFF_PX
            else:
                assert R.chirality_px(P[0]) >= R.MIN_DIFF_PX and T[1]["mirror"] == T[0]["mirror"]
