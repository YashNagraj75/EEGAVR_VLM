import numpy as np
from .rules import _finish, check_twin, _clone
from .scene import index, finalize


def partner_pair(partner_rule, b_pair, state, salt=0, pick="random"):
    # A-partner pair from one B base pair: geometry kept, fills set by the A rule, B state chosen by caller
    base = b_pair["pos"] if state else b_pair["twin"]
    objs = [dict(o, fill=False, role="distractor") for o in base["objects"]]
    rng = np.random.default_rng([b_pair["seed"], 11, int(state), int(salt)])
    if partner_rule == "A1":
        j = int(rng.integers(len(objs)))
        if pick in ("max", "min"):
            sz = [o["size"] for o in objs]
            j = int(np.argmax(sz) if pick == "max" else np.argmin(sz))
        objs[j].update(fill=True, role="t1")
        pos = index(_clone(objs))
        twin = _clone(pos)
        twin[j] = dict(twin[j], fill=False)
        declared, knob = {(j, "fill")}, objs[j]["size"]
    elif partner_rule == "A2":
        objs = [dict(o, fill=b) for o, b in zip(objs, [q["fill"] for q in base["objects"]])]
        tri = [k for k, o in enumerate(objs) if o["shape"] == "triangle"]
        for k in tri:
            objs[k]["role"] = "t1"
        pos = index(_clone(objs))
        twin = [finalize(dict(o, shape="pentagon")) if k in tri else dict(o) for k, o in enumerate(pos)]
        declared, knob = {(k, "shape") for k in tri}, float(len(objs) - len(tri))
    else:
        jt, js = b_pair["meta"]["slot_T"], b_pair["meta"]["slot_S"]
        objs[jt].update(fill=True, role="t1")
        objs[js].update(role="t2")
        pos = index(_clone(objs))
        twin = _clone(pos)
        twin[jt] = dict(twin[jt], fill=False)
        twin[js] = dict(twin[js], fill=True)
        declared, knob = {(jt, "fill"), (js, "fill")}, 0.0
    P, T = _finish(pos, "default"), _finish(twin, "default")
    bad = check_twin(partner_rule, P, T, declared)
    if bad:
        raise RuntimeError(f"partner twin rejected: {bad}")
    return {"pos": P, "twin": T, "knob": float(knob), "seed": b_pair["seed"], "b_state": bool(state),
            "base_setting_id": b_pair["setting_id"], "salt": int(salt), "pick": pick, "draws": 1, "meta": {}}
