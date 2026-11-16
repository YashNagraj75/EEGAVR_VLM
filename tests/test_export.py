import numpy as np
from eegavr.generator.assemble import build_job, regenerate
from eegavr.generator.export import scene_json
from eegavr.generator.render import render
from eegavr.audit.competing import literal_matrix, equivalence_mask
from eegavr.audit.run import make_pair
from eegavr.generator.settings import all_settings


def _eq(sid, n=60):
    st = all_settings()[sid]
    sc, y = [], []
    for i in range(n):
        p = make_pair(st, i)
        sc += [p["pos"], p["twin"]]
        y += [True, False]
    L, _ = literal_matrix(sc)
    return equivalence_mask(L, np.array(y))


def test_problem_structure_and_regeneration():
    eq = {s: _eq(s) for s in ("B1-s01", "B1-s02", "B1-s03", "A1pB1-s01")}
    j = build_job(("B1", "test_id", 1, eq, {}))
    probs = {p["problem_id"]: p for p in j["problems"]}
    b, a = probs["B1-test_id-0001"], probs["A1pB1-test_id-0001"]
    assert b["partner_problem_id"] == a["problem_id"] and a["partner_problem_id"] == b["problem_id"]
    for p in (a, b):
        assert len(p["left"]) == 6 and len(p["right"]) == 6 and len(p["test"]) == 2
        pos = p[p["positive_side"]]
        neg = p["right" if p["positive_side"] == "left" else "left"]
        for k, pid in enumerate(pos):
            assert neg.index(p["twin_map"][pid]) != k
        assert {t["answer"] for t in p["test"]} == {"left", "right"}
        for pid, txt in p["described"].items():
            assert "t1" not in txt and "B1" not in txt and "role" not in txt
    panels = {pid: (sc, seed, rg) for pid, sc, seed, rg in j["panels"]}
    for pid, (sc, seed, rg) in list(panels.items())[:6] + list(panels.items())[-6:]:
        sj = scene_json(sc, pid, seed, rg)
        r = regenerate(sj)
        assert np.array_equal(render(r), render(sc))
