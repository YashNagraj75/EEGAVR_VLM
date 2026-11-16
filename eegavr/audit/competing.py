import numpy as np
from ..generator.predicates import candidate_predicates

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


def equivalence_mask(L_pool, y_pool):
    # candidate ANDs that mean the same as the intended rule on the setting's reference pool
    agree = _pair_counts(L_pool, y_pool) / len(y_pool)
    return (agree >= EQUIV) | (agree <= 1 - EQUIV)


def competing(L_prob, y_prob, eqmask, names):
    # non-equivalent candidates that sort (or anti-sort) all panels of one problem
    cnt = _pair_counts(L_prob, y_prob)
    n = len(y_prob)
    hit = ((cnt == n) | (cnt == 0)) & ~eqmask
    P = len(names)
    lit = names + [f"NOT {x}" for x in names]
    ii, jj = np.nonzero(np.triu(hit))
    return [lit[i] if i == j else f"{lit[i]} AND {lit[j]}" for i, j in zip(ii, jj)]
