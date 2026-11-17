import numpy as np
from ..generator.predicates import candidate_predicates
from ..generator.rules_a import FAMILY_A

EQUIV = 0.99
_CANDS = {}


def cands(extended=True):
    if extended not in _CANDS:
        _CANDS[extended] = candidate_predicates(extended)
    return _CANDS[extended]


def literal_matrix(scenes, extended=True):
    # rows: candidate predicates evaluated on each scene
    C = cands(extended)
    return np.array([[bool(f(s)) for s in scenes] for f in C.values()], bool), list(C)


def _pair_counts(L, y):
    # agreements of AND(Li, Lj) with y for all literal pairs (literals = predicates and their negations)
    Lit = np.vstack([L, ~L]).astype(np.int32)
    yi = y.astype(np.int32)
    tp = (Lit * yi) @ Lit.T
    fp = (Lit * (1 - yi)) @ Lit.T
    return tp + ((1 - yi).sum() - fp)


FAMILY = {"A1": ("A1:",), "A6": ("A6:",), "B1": ("B1:", "B1':"), "B3": ("B3:",), "B4": ("B4:",), "C1": ("C:count",), "C1b": ("C1b:",), "A2": ("A2:",)}
FAMILY.update(FAMILY_A)


def equivalence_mask(L_pool, y_pool, names=None, rule=None):
    # candidates counted as the intended rule (pool-equivalent, an equivalent literal, or own-family/constant literals)
    agree = _pair_counts(L_pool, y_pool) / len(y_pool)
    eq = (agree >= EQUIV) | (agree <= 1 - EQUIV)
    d = np.diag(eq).copy()
    eq = eq | d[:, None] | d[None, :]
    if names is None or rule is None:
        return eq
    fam = np.array([n.startswith(FAMILY[rule]) for n in names])
    frac = L_pool.mean(1)
    const = (frac >= EQUIV) | (frac <= 1 - EQUIV)
    yv = y_pool.astype(bool)
    npos = L_pool.sum(1)
    sub = (npos > 0) & (((L_pool & ~yv).sum(1) <= (1 - EQUIV) * npos) | ((L_pool & yv).sum(1) <= (1 - EQUIV) * npos))
    fam2, const2 = np.r_[fam, fam], np.r_[const | sub, const | sub]
    ok = fam2 | const2
    pair_ok = ok[:, None] & ok[None, :] & (fam2[:, None] | fam2[None, :])
    return {"eq": eq | pair_ok, "own": fam2 | np.diag(eq)}


def competing(L_prob, y_prob, eqmask, names):
    # non-excused candidates sorting all panels; columns are [positives..., their twins...] in the same order
    if isinstance(eqmask, dict):
        own, eqm = eqmask["own"], eqmask["eq"]
        h = L_prob.shape[1] // 2
        pc = np.all(L_prob[:, :h] == L_prob[:, h:], axis=1)
        pc2 = np.r_[pc, pc]
        eqm = eqm | (own[:, None] & pc2[None, :]) | (pc2[:, None] & own[None, :])
    else:
        eqm = eqmask
    cnt = _pair_counts(L_prob, y_prob)
    n = len(y_prob)
    hit = ((cnt == n) | (cnt == 0)) & ~eqm
    P = len(names)
    lit = names + [f"NOT {x}" for x in names]
    ii, jj = np.nonzero(np.triu(hit))
    return [lit[i] if i == j else f"{lit[i]} AND {lit[j]}" for i, j in zip(ii, jj)]
