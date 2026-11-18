from eegavr.audit.run import make_pair
from eegavr.generator.export import _order, symbolic
from eegavr.generator.settings import all_settings

ST = all_settings()


def _has(R, name):
    return [r[1:] for r in R if r[0] == name]


def _decide(rule, sym):
    # the B rule computed from the symbolic relations list and object shapes only
    R, shape = sym["relations"], {o["id"]: o["shape"] for o in sym["objects"]}
    same = {tuple(r) for r in _has(R, "same_shape")}
    if rule == "B2":
        return any(shape[a] == "triangle" and shape[b] == "circle" for a, b in _has(R, "left_of"))
    if rule == "B5":
        return bool(_has(R, "collinear"))
    if rule == "B6":
        return bool(_has(R, "same_orientation"))
    if rule == "B7":
        return bool(_has(R, "mirror_image"))
    if rule == "B8":
        return any(tuple(sorted((a, b))) in same and shape[m] != shape[a] for m, a, b in _has(R, "between"))
    return any(tuple(r) in same for r in _has(R, "closest_pair"))


def test_relations_suffice_for_new_b_rules():
    for rule in ("B2", "B5", "B6", "B7", "B8", "B9"):
        for sid in (f"{rule}-s01", f"{rule}-oc"):
            for i in range(6):
                p = make_pair(ST[sid], i)
                for sc, truth in ((p["pos"], True), (p["twin"], False)):
                    sym = symbolic(sc, _order(f"{sid}-{i}-{truth}", len(sc["objects"])))
                    assert _decide(rule, sym) == truth, (sid, i, truth)


def test_one_closest_pair_per_panel():
    p = make_pair(ST["B1-s01"], 0)
    sym = symbolic(p["pos"], _order("x", len(p["pos"]["objects"])))
    assert len(_has(sym["relations"], "closest_pair")) == 1
