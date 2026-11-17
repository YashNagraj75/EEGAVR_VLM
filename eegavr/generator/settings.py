import copy
from .scene import SIZE_RANGE
from .rules_a import register_settings as register_a
from . import rules_blayout

ID_N = (2, 4)
OOD_N = (7, 8)
SMALL = (0.06, 0.10)
SPLIT_CODE = {"audit": 1, "validation": 2, "test_id": 3, "ood_count": 4, "ood_count_control": 5, "ood_render": 6}
RULE_TYPE = {"A1": "A", "A6": "A", "B1": "B", "B3": "B", "B4": "B", "C1": "C", "C1b": "C"}
V11_RULES = ("C1b",)
PARTNERS = {"B1": "A1", "B3": "A6", "B4": "A1"}
PARTNER_FAILED = {"C1b": "A2"}
NO_OOD_COUNT = ("C1",)
EXTRA_ID, EXTRA_OOD, V20_RULES = {}, {}, []
RULE_PROGRAM = {
    "A1": "exists o: o.fill == filled",
    "A6": "exists o: o.shape == triangle and o.fill == filled",
    "B1": "exists t(triangle), c(circle): c.extent_top - t.extent_bottom > 0  (y from the top)",
    "B3": "exists o, i: o.region contains i.region and dist(i, boundary(o)) >= 0",
    "B4": "exists a, b: ink(a) meets ink(b), regions do not overlap, no nesting in the scene",
    "C1": "count(objects) == 3",
    "C1b": "count(o: o.shape == triangle) == 3  (total count fixed within each twin pair)",
    "A2": "exists o: o.shape == triangle",
}
KNOB_NAME = {"A1": "size of the filled shape", "A6": "number of conjunction distractors",
             "B1": "vertical gap between the extents", "B3": "margin of the inner shape to the outer boundary",
             "B4": "ink gap in the twin", "C1": "count pair (3 vs 4)",
             "C1b": "triangle count pair (3 vs 2) at a fixed total of 5-6 shapes", "A2": "number of non-triangle shapes"}

ID_BINS = {
    "A1": [(0.06, 0.09), (0.09, 0.12), (0.12, 0.16)],
    "A6": [(0, 0), (1, 1), (2, 2)],
    "B1": [(0.03, 0.06), (0.06, 0.12), (0.12, 0.24)],
    "B3": [(0.03, 0.045), (0.045, 0.06), (0.06, 0.08)],
    "B4": [(0.010, 0.015), (0.015, 0.020), (0.020, 0.040)],
    "C1": [(3, 3)],
    "C1b": [(3, 3)],
}
FULL = {"A1": None, "A6": (1, 2), "B1": (0.03, 0.24), "B3": (0.06, 0.08), "B4": (0.010, 0.015), "C1": (3, 3), "C1b": (3, 3)}
TUNING = {
    "C1b": {"n_objects_id": (5, 6)},
    "A6": {"a6_pool": ["FS", "OT", "OS"]},
    "B1": {"role_fill": "outline", "role_ink_jitter": 0.03, "role_ink_offset": 0.005, "n_objects_id": (4, 4)},
    "B3": {"twin_within_bbox": True, "b3_twin_near": 0.06},
    "B4": {"symmetric_separation": True, "stroke_contact_eps_px": 0.25, "n_objects_id": (4, 4),
           "distractors_span_roles": True},
    "C1": {"c1_controls": ["up", "down"], "c1_size_shift": (0.10, 0.30), "c1_size_offset": 0.05,
           "c1_layout_shift": (0.20, 0.50), "c1_layout_offset": 0.10},
}
for _reg in (rules_blayout,):
    V20_RULES += _reg.RULES
    for _name in ("RULE_TYPE", "RULE_PROGRAM", "KNOB_NAME", "ID_BINS", "FULL", "TUNING"):
        globals()[_name].update(getattr(_reg, _name))


def base(rule, sid, knob, n=ID_N, size_range=SIZE_RANGE, style="default"):
    st = {"id": sid, "rule": rule, "type": RULE_TYPE[rule], "knob": tuple(knob), "knob_name": KNOB_NAME[rule],
          "n_objects": tuple(n), "size_range": tuple(size_range), "style": style}
    if rule == "B3":
        st["n_objects"] = (4, 4) if n == ID_N else (n[0], n[1])
        st["b3_outer_range"] = SIZE_RANGE
    if rule == "A6":
        st["n_objects"] = None
    tune = copy.deepcopy(TUNING.get(rule, {}))
    if "n_objects_id" in tune:
        n_id = tune.pop("n_objects_id")
        if n == ID_N:
            st["n_objects"] = tuple(n_id)
    st.update(tune)
    return st


DROPPED = {"A6-s01": "5c: no distractor means 'outline square present' sorts every problem"}
MARGINAL_MAX_5A = 0.555
MARGINAL_MIN_5D = 0.999


def audit_tier(rep):
    # pass / marginal (user rule 2026-11-16: 5a pass-probe <= 0.555 and 5d >= 99.9%) / excluded, from an audit JSON
    if rep["setting_id"] in DROPPED or rep.get("setting", {}).get("partner_of") in PARTNER_FAILED:
        return "excluded"
    d = rep["5d"].get("224", rep["5d"].get(224))["all_ok"]
    a = rep["5a"]["pass_set"]
    if rep["type"] == "C":
        u = rep["c_univariate"]
        ok = u["pixel"]["pass"] and u["symbolic"]["pass"] and d == 1.0
        if ok and a["pass"]:
            return "pass"
        return "marginal" if ok and a["best_acc"] <= MARGINAL_MAX_5A else "excluded"
    if rep["pass"]:
        return "pass"
    if rep["type"] == "B" and rep["pass_5b"] and a["best_acc"] <= MARGINAL_MAX_5A and d >= MARGINAL_MIN_5D:
        return "marginal"
    return "excluded"


def id_settings(rule):
    if rule in EXTRA_ID:
        return [dict(s) for s in EXTRA_ID[rule]]
    return [base(rule, f"{rule}-s{i + 1:02d}", b) for i, b in enumerate(ID_BINS[rule])]


def active_id_settings(rule):
    return [st for st in id_settings(rule) if st["id"] not in DROPPED]


def ood_settings(rule):
    if rule in EXTRA_OOD:
        return {k: dict(v) for k, v in EXTRA_OOD[rule].items()}
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
    # v1 settings first and v1.1 rules last, so setting codes (and seeds) of v1 settings never change
    S = {}
    for r in [r for r in RULE_TYPE if r not in V11_RULES and r not in V20_RULES]:
        for st in id_settings(r) + list(ood_settings(r).values()):
            S[st["id"]] = st
    later = {b: a for b, a in PARTNERS.items() if b in V11_RULES}
    for b, a in PARTNERS.items():
        if b not in later:
            S[f"{a}p{b}-s01"] = _partner(a, b)
    for r in V11_RULES:
        for st in id_settings(r) + list(ood_settings(r).values()):
            S[st["id"]] = st
    for b, a in later.items():
        S[f"{a}p{b}-s01"] = _partner(a, b)
    for r in V20_RULES:
        for st in id_settings(r) + list(ood_settings(r).values()):
            S[st["id"]] = st
    return S


def _partner(a, b):
    kn = {"A1": KNOB_NAME["A1"], "A2": KNOB_NAME["A2"]}.get(a, "derived from the B3 base scene")
    return {"id": f"{a}p{b}-s01", "rule": a, "type": "A", "partner_of": b, "knob": None, "knob_name": kn,
            "style": "default"}


register_a(globals())
SETTING_CODE = {sid: i + 1 for i, sid in enumerate(all_settings())}


def seed_for(split, sid, counter):
    # disjoint seed blocks: split code x 1e9 + setting code x 1e6 + counter
    assert counter < 10**6
    return SPLIT_CODE[split] * 10**9 + SETTING_CODE[sid] * 10**6 + counter


def b_setting_for(b_rule, i):
    act = active_id_settings(b_rule)
    return act[i % len(act)]
