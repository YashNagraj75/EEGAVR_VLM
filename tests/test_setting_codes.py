import random
from eegavr.generator import settings as S

BLOCKS = {"A2": 100, "A3": 120, "A4": 140, "A5": 160, "B2": 200, "B5": 220, "B6": 240, "B7": 260, "B8": 280, "B9": 300,
          "C2b": 700, "C3b": 704, "C4": 708, "C4b": 714}
SUFFIX6 = ("s01", "s02", "s03", "oc", "occ", "or")
SUFFIX4 = ("s01", "oc", "occ", "or")


def test_v20_code_blocks_pinned():
    assert S.CODE_BLOCK == BLOCKS
    for r, base in BLOCKS.items():
        suf = SUFFIX4 if r in ("C2b", "C3b") else SUFFIX6
        for k, x in enumerate(suf):
            assert S.SETTING_CODE[f"{r}-{x}"] == base + k


def test_pilot_codes_are_1_to_40():
    pilot = {sid: c for sid, c in S.SETTING_CODE.items() if c <= 40}
    assert sorted(pilot.values()) == list(range(1, 41))
    assert S.SETTING_CODE["A1-s01"] == 1 and S.SETTING_CODE["A1pB4-s01"] == 36 and S.SETTING_CODE["C1b-or"] == 40


def test_codes_independent_of_registration_order():
    ref = dict(S.SETTING_CODE)
    order = list(S.V20_RULES)
    try:
        for seed in range(3):
            random.Random(seed).shuffle(S.V20_RULES)
            assert S.setting_codes(S.all_settings()) == ref
    finally:
        S.V20_RULES[:] = order


def test_codes_unique_and_below_split_stride():
    v = list(S.SETTING_CODE.values())
    assert len(set(v)) == len(v) and max(v) < 1000
