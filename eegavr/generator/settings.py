import copy
from .scene import SIZE_RANGE

ID_N = (2, 4)
OOD_N = (7, 8)
SMALL = (0.06, 0.10)
SPLIT_CODE = {"audit": 1, "validation": 2, "test_id": 3, "ood_count": 4, "ood_count_control": 5, "ood_render": 6}
RULE_TYPE = {"A1": "A", "A6": "A", "B1": "B", "B3": "B", "B4": "B", "C1": "C"}
PARTNERS = {"B1": "A1", "B3": "A6", "B4": "A1"}
RULE_PROGRAM = {
    "A1": "exists o: o.fill == filled",
    "A6": "exists o: o.shape == triangle and o.fill == filled",
    "B1": "exists t(triangle), c(circle): c.extent_top - t.extent_bottom > 0  (y from the top)",
    "B3": "exists o, i: o.region contains i.region and dist(i, boundary(o)) >= 0",
    "B4": "exists a, b: ink(a) meets ink(b), regions do not overlap, no nesting in the scene",
    "C1": "count(objects) == 3",
}
KNOB_NAME = {"A1": "size of the filled shape", "A6": "number of conjunction distractors",
             "B1": "vertical gap between the extents", "B3": "margin of the inner shape to the outer boundary",
             "B4": "ink gap in the twin", "C1": "count pair (3 vs 4)"}

ID_BINS = {
    "A1": [(0.06, 0.09), (0.09, 0.12), (0.12, 0.16)],
    "A6": [(0, 0), (1, 1), (2, 2)],
    "B1": [(0.03, 0.06), (0.06, 0.12), (0.12, 0.24)],
    "B3": [(0.03, 0.045), (0.045, 0.06), (0.06, 0.08)],
    "B4": [(0.01, 0.02), (0.02, 0.04), (0.04, 0.07)],
    "C1": [(3, 3)],
}
FULL = {"A1": None, "A6": (1, 2), "B1": (0.03, 0.24), "B3": (0.03, 0.08), "B4": (0.01, 0.07), "C1": (3, 3)}
TUNING = {}


def base(rule, sid, knob, n=ID_N, size_range=SIZE_RANGE, style="default"):
    st = {"id": sid, "rule": rule, "type": RULE_TYPE[rule], "knob": tuple(knob), "knob_name": KNOB_NAME[rule],
          "n_objects": tuple(n), "size_range": tuple(size_range), "style": style}
    if rule == "B3":
        st["n_objects"] = (4, 4) if n == ID_N else (n[0], n[1])
        st["b3_outer_range"] = SIZE_RANGE
    if rule == "A6":
        st["n_objects"] = None
    st.update(copy.deepcopy(TUNING.get(rule, {})))
    return st


def id_settings(rule):
    return [base(rule, f"{rule}-s{i + 1:02d}", b) for i, b in enumerate(ID_BINS[rule])]


def ood_settings(rule):
    out = {}
    full = FULL[rule]
    if rule == "A1":
        out["ood_count"] = base(rule, "A1-oc", SMALL, OOD_N, SMALL)
        out["ood_count_control"] = base(rule, "A1-occ", SMALL, ID_N, SMALL)
        out["ood_render"] = base(rule, "A1-or", SIZE_RANGE, ID_N, SIZE_RANGE, "hatch")
        return out
    if rule == "A6":
        out["ood_count"] = base(rule, "A6-oc", (5, 6), OOD_N, SMALL)
    elif rule != "C1":
        out["ood_count"] = base(rule, f"{rule}-oc", full, OOD_N, SMALL)
    out["ood_count_control"] = base(rule, f"{rule}-occ", full, ID_N, SMALL)
    out["ood_render"] = base(rule, f"{rule}-or", full, ID_N, SIZE_RANGE, "hatch")
    return out


def all_settings():
    S = {}
    for r in RULE_TYPE:
        for st in id_settings(r):
            S[st["id"]] = st
        for st in ood_settings(r).values():
            S[st["id"]] = st
    for b, a in PARTNERS.items():
        S[f"{a}p{b}-s01"] = {"id": f"{a}p{b}-s01", "rule": a, "type": "A", "partner_of": b, "knob": None,
                             "knob_name": KNOB_NAME[a] if a == "A1" else "derived from the B3 base scene",
                             "style": "default"}
    return S


SETTING_CODE = {sid: i + 1 for i, sid in enumerate(all_settings())}


def seed_for(split, sid, counter):
    # disjoint seed blocks: split code x 1e9 + setting code x 1e6 + counter
    assert counter < 10**6
    return SPLIT_CODE[split] * 10**9 + SETTING_CODE[sid] * 10**6 + counter


def b_setting_for(b_rule, i):
    return id_settings(b_rule)[i % len(ID_BINS[b_rule])]
