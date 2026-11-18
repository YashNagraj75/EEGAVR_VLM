import numpy as np
from eegavr.audit.competing import competing


def test_own_literal_conjunction_is_excused_but_foreign_pair_is_not():
    y = np.array([True] * 7 + [False] * 7)
    own_lit = y.copy()
    rare = np.zeros(14, bool)
    a = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1], bool)
    b = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], bool)
    L = np.vstack([own_lit, rare, a, b])
    names = ["R:own", "X:rare", "X:a", "X:b"]
    own = np.r_[[True, False, False, False], [True, False, False, False]]
    eq = np.zeros((8, 8), bool)
    eq[0, 0] = eq[4, 4] = True
    out = competing(L, y, {"eq": eq, "own": own}, names)
    assert "R:own AND NOT X:rare" not in out
    assert "X:a AND X:b" in out
