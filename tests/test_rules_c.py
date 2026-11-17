import numpy as np
import pytest
from eegavr.generator.scene import make_obj, index, ink_area
from eegavr.generator import rules_c as RC
from eegavr.generator.predicates import RULE_FN, candidate_predicates
from eegavr.generator.rules import sample_pair
from eegavr.generator.settings import all_settings, seed_for, SETTING_CODE, RULE_TYPE
from eegavr.generator.render import render
from shapely.ops import unary_union
from eegavr.generator.scene import poly

ST = all_settings()
C_SIDS = [s for s in ST if s.split("-")[0] in RC.C_RULES]


def sc(*objs):
    return {"objects": index(list(objs))}


def test_c_predicates():
    circ = [make_obj("circle", .06, False, .15 + .17 * k, .5, 0, 4) for k in range(5)]
    assert RC.c2b(sc(*circ[:4])) and not RC.c2b(sc(*circ)) and not RC.c2b(sc(*circ[:3]))
    sq = [make_obj("square", .06, False, .15 + .17 * k, .5, 0, 4) for k in range(5)]
    assert RC.c3b(sc(*sq)) and not RC.c3b(sc(*sq[:4]))
    f = [dict(o, fill=k < 2) for k, o in enumerate(circ[:4])]
    assert RC.c4(sc(*f)) and not RC.c4(sc(*[dict(o, fill=True) for o in f]))
    assert RC.c4b(sc(*f)) and not RC.c4b(sc(*[dict(o, fill=k < 1) for k, o in enumerate(f)]))
    for r in RC.C_RULES:
        assert RULE_FN[r] is RC.C_RULE_FN[r] and RULE_TYPE[r] == "C"


def test_c_settings_use_fixed_code_block():
    assert all(SETTING_CODE[s] >= RC.C_CODE_BASE for s in C_SIDS)
    assert all(SETTING_CODE[s] < RC.C_CODE_BASE for s in ST if s not in C_SIDS)
    assert SETTING_CODE["C1b-s01"] == 37 and SETTING_CODE["C1b-or"] == 40 and SETTING_CODE["A1pB4-s01"] == 36
    assert ST["C3b-s01"]["size_range"] == (0.06, 0.12) and ST["C3b-occ"]["size_range"] == (0.06, 0.10)
    assert ST["C3b-oc"]["n_objects"] == (7, 8) and ST["C4-s02"]["n_objects"] is None


def test_c_candidates_registered():
    C = candidate_predicates()
    assert "C2b:circle count==4" in C and "C3b:square count>=5" in C and "C4b:filled circle count==2" in C


@pytest.mark.parametrize("sid", C_SIDS)
def test_c_pairs_valid_and_deterministic(sid):
    st = ST[sid]
    for i in range(2):
        a = sample_pair(st, seed_for("audit", sid, i))
        b = sample_pair(st, seed_for("audit", sid, i))
        assert a["pos"]["objects"] == b["pos"]["objects"] and a["twin"]["objects"] == b["twin"]["objects"]
        assert a["pos"]["truth"][st["rule"]] and not a["twin"]["truth"][st["rule"]]
        assert len(a["pos"]["objects"]) == len(a["twin"]["objects"])
        assert np.array_equal(render(a["twin"]), render(b["twin"]))


@pytest.mark.parametrize("sid", ["C2b-s01", "C3b-s01", "C4b-s02"])
def test_count_by_property_twins_keep_ink(sid):
    for i in range(3):
        p = sample_pair(ST[sid], seed_for("audit", sid, i))
        ip, it = (sum(ink_area(o) for o in p[x]["objects"]) for x in ("pos", "twin"))
        assert abs(it - ip) / ip <= 0.01


def test_c3b_interior_target_keeps_hull():
    for i in range(3):
        p = sample_pair(ST["C3b-s01"], seed_for("audit", "C3b-s01", i))
        h = [unary_union([poly(o) for o in p[x]["objects"]]).convex_hull.area for x in ("pos", "twin")]
        assert abs(h[0] - h[1]) < 1e-9


def test_edge_matched_replacement_candidates():
    m = RC.matched_replacements("circle", 0.0, RC.COUNT_REPLACE["C2b"])
    assert set(m) == set(RC.COUNT_REPLACE["C2b"])
    for sh, rots in m.items():
        d = np.abs(RC.edge_table(sh)[rots] / RC.edge_table("circle")[0] - 1).max(1)
        assert np.all(d <= RC.MATCH_TOL)
