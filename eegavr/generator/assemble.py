import hashlib
import numpy as np
from .settings import all_settings, seed_for, SPLIT_CODE, SETTING_CODE, PARTNERS, active_id_settings, ood_settings
from .rules import sample_pair
from .partners import partner_pair
from ..audit.competing import literal_matrix, competing

N_SUPPORT = 6
MAX_ATTEMPTS = 14
PARTNER_TRIES = 20


def panel_id(sid, split, counter, member, extra=""):
    # opaque id: setting prefix plus a hash that does not reveal which side the panel is on
    h = hashlib.sha1(f"{split}|{sid}|{counter}|{member}|{extra}".encode()).hexdigest()[:10]
    return f"{sid}-{h}"


def _derangement(rng, n):
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def _layout(rng, pairs, rule, ids):
    # positives on a random side, twins never in the matching display position, one test panel per side
    side = "left" if rng.random() < 0.5 else "right"
    other = "right" if side == "left" else "left"
    order = rng.permutation(N_SUPPORT)
    der = _derangement(rng, N_SUPPORT)
    pos_ids = [ids[k][0] for k in order]
    neg_ids = [None] * N_SUPPORT
    for disp, k in enumerate(order):
        neg_ids[der[disp]] = ids[k][1]
    test = [{"panel_id": ids[N_SUPPORT][0], "answer": side}, {"panel_id": ids[N_SUPPORT][1], "answer": other}]
    if rng.random() < 0.5:
        test = test[::-1]
    out = {"positive_side": side, side: pos_ids, other: neg_ids, "test": test,
           "twin_map": {ids[k][0]: ids[k][1] for k in range(N_SUPPORT + 1)}}
    return out


def _scenes_labels(pairs):
    sc = [p["pos"] for p in pairs] + [p["twin"] for p in pairs]
    y = np.array([True] * len(pairs) + [False] * len(pairs))
    return sc, y


def check_5c(pairs, eq):
    sc, y = _scenes_labels(pairs)
    L, names = literal_matrix(sc)
    for s in sc:
        s.pop("_polys", None)
    return competing(L, y, eq, names)


def problem_setting(rule, split, k):
    if split in ("validation", "test_id"):
        bins = active_id_settings(rule)
        return bins[k % len(bins)]
    return ood_settings(rule)[split]


def build_problem(rule, split, k, eqmasks, stats, tuning=None, start=0):
    # one problem of a rule: 7 twin pairs of one setting, 5c-checked, resampled until no competing rule
    st = dict(problem_setting(rule, split, k))
    st.update((tuning or {}).get(rule, {}))
    sid = st["id"]
    ctrl = ("equal_ink" if k % 2 else "equal_size") if rule == "C1" else None
    for att in range(start, MAX_ATTEMPTS):
        pairs = []
        for j in range(N_SUPPORT + 1):
            c = k * 100 + att * 7 + j
            p = sample_pair(st, seed_for(split, sid, c), control=ctrl)
            p["counter"], p["split"] = c, split
            p["meta"]["control"] = ctrl
            pairs.append(p)
        stats["draws"] += sum(p["draws"] for p in pairs)
        stats["pairs"] += len(pairs)
        comp = check_5c(pairs, eqmasks[sid])
        stats["attempts"] += 1
        if comp:
            stats["rejected"] += 1
            stats["examples"].append(comp[:3])
            continue
        return st, pairs, att
    raise RuntimeError(f"{rule} {split} {k}: 5c failed {MAX_ATTEMPTS} times")


def build_partner(b_rule, pairs, eq, rng, stats):
    # A partner from the same 7 base pairs; B state balanced 3/3 on the support pairs, random on the test pair
    a_rule = PARTNERS[b_rule]
    for salt in range(PARTNER_TRIES):
        states = list(rng.permutation([True] * 3 + [False] * 3)) + [bool(rng.random() < 0.5)]
        picks = ["random"] * len(pairs)
        a, b = rng.choice(len(pairs), 2, replace=False)
        picks[a], picks[b] = "max", "min"
        pp = [partner_pair(a_rule, bp, s, salt, pk) for bp, s, pk in zip(pairs, states, picks)]
        for q, bp in zip(pp, pairs):
            q["counter"], q["split"] = bp["counter"], bp["split"]
        comp = check_5c(pp, eq)
        stats["attempts"] += 1
        if not comp:
            return pp
        stats["rejected"] += 1
        stats["examples"].append(comp[:3])
    return None


def _panel_entries(p, sid, split, member_ids, regen_extra=None):
    out = []
    for sc, pid in ((p["pos"], member_ids[0]), (p["twin"], member_ids[1])):
        regen = {"setting_id": sid, "split": split, "counter": p["counter"], "control": p["meta"].get("control")}
        if regen_extra:
            regen.update(regen_extra)
        out.append((pid, sc, p["seed"], regen))
    return out


def build_job(args):
    # all problems of one rule x split (plus A partners for B rules in test_id)
    from .export import described, symbolic, _order
    rule, split, n, eqmasks, tuning = args
    stats = {"draws": 0, "pairs": 0, "attempts": 0, "rejected": 0, "examples": []}
    pstats = {"attempts": 0, "rejected": 0, "examples": []}
    problems, panels = [], []
    for k in range(n):
        start = 0
        while True:
            st, pairs, att = build_problem(rule, split, k, eqmasks, stats, tuning, start)
            partner = None
            if split == "test_id" and rule in PARTNERS:
                psid = f"{PARTNERS[rule]}p{rule}-s01"
                rng = np.random.default_rng([SPLIT_CODE[split], SETTING_CODE[psid], k, att])
                partner = build_partner(rule, pairs, eqmasks[psid], rng, pstats)
                if partner is None:
                    start = att + 1
                    continue
            break
        sid = st["id"]
        groups = [(rule, sid, pairs, f"{rule}-{split}-{k + 1:04d}", None)]
        if partner is not None:
            groups.append((PARTNERS[rule], f"{PARTNERS[rule]}p{rule}-s01", partner,
                           f"{PARTNERS[rule]}p{rule}-{split}-{k + 1:04d}", sid))
        pids = [g[3] for g in groups]
        for gi, (r, gsid, prs, prob_id, base_sid) in enumerate(groups):
            ids = []
            for p in prs:
                extra = None
                if base_sid:
                    extra = {"base_setting_id": p["base_setting_id"], "base_seed": p["seed"], "salt": p["salt"],
                             "pick": p["pick"]}
                tag = f"{p.get('b_state')}{p.get('salt')}" if base_sid else ""
                mid = (panel_id(gsid, split, p["counter"], "p", tag), panel_id(gsid, split, p["counter"], "n", tag))
                ids.append(mid)
                panels += _panel_entries(p, gsid, split, mid, extra)
            rng = np.random.default_rng([SPLIT_CODE[split], SETTING_CODE[gsid], k, 7])
            lay = _layout(rng, prs, r, ids)
            allp = {pid: sc for p, mid in zip(prs, ids) for pid, sc in zip(mid, (p["pos"], p["twin"]))}
            desc, sym = {}, {}
            for pid, sc in allp.items():
                o = _order(pid, len(sc["objects"]))
                desc[pid] = described(sc, o)
                sym[pid] = symbolic(sc, o)
            prob = {"problem_id": prob_id, "rule_id": r, "type": "A" if r.startswith("A") else r[0],
                    "partner_problem_id": pids[1 - gi] if len(pids) == 2 else None, "rule_setting_id": gsid,
                    "knob": float(np.mean([p["knob"] for p in prs])), "split": split,
                    "positive_side": lay["positive_side"], "left": lay["left"], "right": lay["right"],
                    "test": lay["test"], "twin_map": lay["twin_map"], "described": desc, "symbolic": sym}
            problems.append(prob)
    return {"rule": rule, "split": split, "problems": problems, "panels": panels, "stats": stats,
            "partner_stats": pstats}


def regenerate(scene_json):
    # rebuild a panel's scene from its seed and regen record; returns the matching scene dict
    rg = scene_json["regen"]
    S = all_settings()
    if "base_setting_id" in rg:
        bst = S[rg["base_setting_id"]]
        bp = sample_pair(bst, rg["base_seed"])
        cands = []
        for state in (True, False):
            pp = partner_pair(S[rg["setting_id"]]["rule"], bp, state, rg["salt"], rg.get("pick", "random"))
            cands += [pp["pos"], pp["twin"]]
    else:
        p = sample_pair(S[rg["setting_id"]], scene_json["seed"], control=rg["control"])
        cands = [p["pos"], p["twin"]]
    keys = ("shape", "fill", "x", "y", "rotation", "size")
    for c in cands:
        if len(c["objects"]) == len(scene_json["objects"]) and all(
                a["shape"] == b["shape"] and a["fill"] == b["fill"] and abs(a["x"] - b["x"]) < 1e-5
                and abs(a["y"] - b["y"]) < 1e-5 and abs(a["size"] - b["size"]) < 1e-5
                for a, b in zip(c["objects"], scene_json["objects"])):
            return c
    raise ValueError("no regenerated scene matches")
