from eegavr.generator import rules_a as RA
from eegavr.generator.partners import partner_pair
from eegavr.generator.partners_v2 import V20_PARTNERS
from eegavr.generator.predicates import RULE_FN
from eegavr.generator.rules import sample_pair
from eegavr.generator.scene import layout_ok
from eegavr.generator.settings import all_settings, b_setting_for, seed_for, SETTING_CODE

ST = all_settings()


def test_v20_partner_settings_registered():
    for b, a in V20_PARTNERS.items():
        sid = f"{a}p{b}-s01"
        assert ST[sid]["partner_of"] == b and ST[sid]["rule"] == a and 400 <= SETTING_CODE[sid] <= 405


def test_partner_pairs_flip_only_the_a_rule():
    for b, a in V20_PARTNERS.items():
        built = 0
        for i in range(12):
            bst = b_setting_for(b, i)
            bp = sample_pair(bst, seed_for("audit", bst["id"], i))
            state = i % 2 == 0
            q = partner_pair(a, bp, state)
            if q is None:
                continue
            built += 1
            P, T = q["pos"]["objects"], q["twin"]["objects"]
            assert RULE_FN[a](q["pos"]) and not RULE_FN[a](q["twin"])
            assert RULE_FN[b](q["pos"]) == state
            assert RULE_FN[b](q["twin"]) == (state and b != "B2")
            assert layout_ok(P) and layout_ok(T) and abs(RA.min_gap(P) - RA.min_gap(T)) < 1e-9
            assert [(o["x"], o["y"]) for o in P] == [(o["x"], o["y"]) for o in T]
        assert built >= 6, b


def test_failed_partner_not_assembled():
    from eegavr.generator.settings import ALL_PARTNERS, PARTNER_FAILED, audit_tier
    assert "B2" not in ALL_PARTNERS and PARTNER_FAILED["B2"] == "A2"
    rep = {"setting_id": "A2pB2-s01", "setting": {"partner_of": "B2"}}
    assert audit_tier(rep) == "excluded"
