import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from eegavr.eval import scoring
from eegavr.eval.budget import estimate_stage
from eegavr.eval.dataset import EvalSet, LeakageError, prompt_view, select_problems, validate_problem
from eegavr.eval.models import internvl, qwen3vl
from eegavr.eval.models.base import Generation
from eegavr.eval.models.mock import MALFORMED, MockVLM
from eegavr.eval.parse import parse_answer
from eegavr.eval.prompts import build_prompt, label_map
from eegavr.eval.runner import Stage, run_stage, stage_log_path
from eegavr.eval.vision import (SPECS, composite_geometry, internvl_grid, layout_summary, prepare_separate,
                                qwen_smart_resize)

REPO = Path(__file__).resolve().parents[1]
STAGES = ["visual:separate", "visual:composite", "described", "symbolic", "symbolic_complete"]


def _all_ids(es):
    return [pid for s in es.splits() for pid in es.split_ids(s)]


def test_mockset_is_contract_conformant(mockset):
    es = EvalSet(mockset)
    assert es.manifest["contract_version"] == 1
    ids = _all_ids(es)
    assert len(ids) == 3 * 16
    for pid in ids:
        prob = es.load_problem(pid)
        for panel in validate_problem(prob):
            im = es.load_panel(panel)
            assert im.size == (512, 512) and im.mode == "L"
            assert (mockset / "masks" / f"{panel}.png").exists() and (mockset / "scenes" / f"{panel}.json").exists()


def test_leak_guards(mockset):
    es = EvalSet(mockset)
    prob = json.loads(json.dumps(es.load_problem("B1-test_id-0000")))
    pid = prob["left"][0]
    bad = json.loads(json.dumps(prob))
    bad["symbolic"][pid]["objects"][0]["role"] = "t1"
    with pytest.raises(LeakageError):
        prompt_view(bad, need_symbolic=True)
    bad = json.loads(json.dumps(prob))
    bad["described"][pid] += f" ({prob['right'][0]})"
    with pytest.raises(LeakageError):
        prompt_view(bad, need_described=True)
    view = prompt_view(prob, True, True)
    assert set(vars(view)) == {"left", "right", "test", "described", "symbolic"}


@pytest.mark.parametrize("spec_key", ["qwen3vl", "internvl"])
@pytest.mark.parametrize("stage", STAGES)
def test_prompt_structure_and_no_leaks(mockset, spec_key, stage):
    es, spec = EvalSet(mockset), SPECS[spec_key]
    st = Stage.parse(f"{stage}:test_id")
    for pid in es.split_ids("test_id"):
        prob = es.load_problem(pid)
        pr = build_prompt(prob, st.condition, st.layout, spec, es.load_panel, es.load_scene)
        text = pr.text_with_placeholders("<image>")
        labels = [f"L{i}" for i in range(1, 7)] + [f"R{i}" for i in range(1, 7)] + ["T1", "T2"]
        pos = [text.find(f"{lab}:") if st.layout != "composite" else text.find(lab) for lab in labels]
        if st.layout != "composite":
            assert all(p >= 0 for p in pos) and pos == sorted(pos), "labels missing or out of order"
        forbidden = [prob["problem_id"], *validate_problem(prob), "positive", "twin", "rule_id", "knob", "partner"]
        for f in forbidden:
            assert f not in text, f"leak: {f}"
        for rid in ("A1", "A6", "B1", "B3", "B4", "C1", "A1p"):
            assert not re.search(rf"\b{rid}\b", text), f"rule id {rid} in prompt"
        assert '"T1": "left|right"' in text
        n = len(pr.images)
        if st.layout == "separate":
            assert n == 14 and all(im.size == (spec.separate_panel_px,) * 2 for im in pr.images)
            lm = label_map(prompt_view(prob))
            ref = np.asarray(prepare_separate(es.load_panel(lm["T2"]), spec))
            assert np.array_equal(np.asarray(pr.images[13]), ref), "T2 image not in last position"
            ref = np.asarray(prepare_separate(es.load_panel(lm["L1"]), spec))
            assert np.array_equal(np.asarray(pr.images[0]), ref)
        elif st.layout == "composite":
            assert n == 1
            want = {"qwen3vl": (2656, 1632), "internvl": (1792, 1344)}[spec_key]
            assert pr.images[0].size == want
        else:
            assert n == 0
            assert text.count("\nL1: ") == 1


def test_composite_panel_placement(mockset):
    es, spec = EvalSet(mockset), SPECS["internvl"]
    prob = es.load_problem("A1-test_id-0001")
    im = build_prompt(prob, "visual", "composite", spec, es.load_panel).images[0]
    geo = composite_geometry(spec.composite_panel_px)
    ox, oy = (im.size[0] - geo["W"]) // 2, (im.size[1] - geo["H"]) // 2
    lm = label_map(prompt_view(prob))
    p = spec.composite_panel_px
    for lab in ("L1", "R6", "T1", "T2"):
        x, y = geo["boxes"][lab]
        crop = np.asarray(im.crop((ox + x, oy + y, ox + x + p, oy + y + p)).convert("L"), dtype=int)
        ref = np.asarray(es.load_panel(lm[lab]).resize((p, p), 4), dtype=int)  # 4 = BOX
        assert np.abs(crop - ref).max() <= 1, lab
    t1x, t2x = geo["boxes"]["T1"][0], geo["boxes"]["T2"][0]
    assert abs((t1x + t2x + p) / 2 - geo["W"] / 2) <= 1, "test panels not centred"


def test_vision_geometry_and_tiling():
    assert internvl_grid(1792, 1344) == (4, 3) and internvl_grid(1792, 1344, vllm_rule=False) == (4, 3)
    assert internvl_grid(448, 448) == (1, 1) and internvl_grid(448, 448, vllm_rule=False) == (1, 1)
    assert qwen_smart_resize(512, 512) == (512, 512)
    assert qwen_smart_resize(1632, 2656) == (1632, 2656)
    q, i = layout_summary(SPECS["qwen3vl"]), layout_summary(SPECS["internvl"])
    assert [q["separate"]["effective_panel_px"], q["composite"]["effective_panel_px"]] == [512, 384]
    assert [i["separate"]["effective_panel_px"], i["composite"]["effective_panel_px"]] == [448, 256]
    assert q["separate"]["visual_tokens"] == 14 * (16 * 16 + 2)
    assert i["composite"]["visual_tokens"] == 13 * 256 + 2


PARSE_CASES = [
    ('{"T1": "left", "T2": "right", "rule": "a"}', "left", "right", "json"),
    ('Sure.\n```json\n{"T1":"Right","T2":"Left","rule":"b"}\n```', "right", "left", "json"),
    ("{'T1': 'left', 'T2': 'left',}", "left", "left", "json_repaired"),
    ('{T1: "right", T2: "left", rule: "c"}', "right", "left", "json_repaired"),
    ("T1 belongs to the left group and T2 goes to the right.", "left", "right", "regex"),
    ("T1: RIGHT\nT2: LEFT", "right", "left", "regex"),
    ('{"T1": "left|right", "T2": "left|right", "rule": "<one sentence>"}', None, None, "unparsable"),
    ('{"T1": "left", "rule": "x"}', "left", None, "partial"),
    ("", None, None, "unparsable"),
    ("I cannot tell.", None, None, "unparsable"),
    ('<think>T1 right</think>{"T1": "left", "T2": "right"}', "left", "right", "json"),
    ('first {"T1": "right", "T2": "right"} final answer {"T1": "left", "T2": "right"}', "left", "right", "json"),
    ('{"test1": "the left group", "test2": "right side"}', "left", "right", "json"),
    ('{"T1": "left", "T2": "right", "rule": "truncat', "left", "right", "regex"),
    ("T1 is left, T1 is right, T2 is left", None, "left", "partial"),
]


@pytest.mark.parametrize("raw,t1,t2,status", PARSE_CASES)
def test_parse(raw, t1, t2, status):
    got = parse_answer(raw)
    assert (got["T1"], got["T2"], got["status"]) == (t1, t2, status)


def test_wilson_and_binomial():
    lo, hi = scoring.wilson(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3
    assert scoring.wilson(0, 10)[0] == 0.0 and abs(scoring.wilson(0, 10)[1] - 0.2775) < 1e-3
    assert abs(scoring.binom_sf(60, 100, 0.5) - 0.02844) < 1e-4
    assert scoring.binom_sf(0, 10, 0.5) == 1.0


def _oracle(es):
    return {pid: [t["answer"] for t in es.load_problem(pid)["test"]] for pid in _all_ids(es)}


def _rows(out, level="stage"):
    _, summary, pairs = scoring.score_dir(out)
    return [r for r in summary if r["level"] == level], pairs


def test_oracle_and_constant_models(mockset, tmp_path):
    es = EvalSet(mockset)
    for mode, out in (("oracle", tmp_path / "o"), ("left", tmp_path / "l")):
        model = MockVLM(mode, answers=_oracle(es))
        for s in STAGES:
            run_stage(model, es, Stage.parse(f"{s}:test_id"), out, batch_size=5, log=lambda *_: None)
    rows, pairs = _rows(tmp_path / "o")
    assert len(rows) == len(STAGES) and all(r["panel_acc"] == 1.0 and r["both_acc"] == 1.0 for r in rows)
    assert len(pairs) == len(STAGES) and all(p["n_pairs"] == 4 and p["panel_diff_B_minus_A"] == 0 for p in pairs)
    rows, _ = _rows(tmp_path / "l")
    assert all(r["panel_acc"] == 0.5 and r["both_acc"] == 0.0 and r["frac_left"] == 1.0 and r["frac_same_side"] == 1.0
               for r in rows)
    trows, _ = _rows(tmp_path / "o", "type")
    assert sorted({r["type"] for r in trows}) == ["A", "B", "C"]
    assert {r["type"]: r["n_puzzles"] for r in trows if r["condition"] == "symbolic"} == {"A": 8, "B": 4, "C": 4}


def test_random_mock_is_deterministic(mockset, tmp_path):
    es = EvalSet(mockset)
    outs = []
    for k in range(2):
        run_stage(MockVLM("random", seed=3), es, Stage.parse("described:test_id"), tmp_path / str(k), log=lambda *_: None)
        outs.append([json.loads(x)["raw_output"] for x in open(stage_log_path(tmp_path / str(k), "mock-random",
                                                                              Stage.parse("described:test_id")))])
    assert outs[0] == outs[1]


def test_malformed_outputs_counted(mockset, tmp_path):
    es = EvalSet(mockset)
    run_stage(MockVLM("malformed"), es, Stage.parse("symbolic:test_id"), tmp_path, batch_size=4, log=lambda *_: None)
    rows, _ = _rows(tmp_path)
    expect = {}
    for k in range(16):
        s = parse_answer(MALFORMED[k % len(MALFORMED)])["status"]
        expect[s] = expect.get(s, 0) + 1
    got = {s: rows[0][f"n_{s}"] for s in ("json", "json_repaired", "regex", "partial", "unparsable")}
    assert got == {s: expect.get(s, 0) for s in got}
    assert expect.get("unparsable", 0) >= 3 and expect.get("partial", 0) >= 1


def test_resume_after_crash_and_truncated_line(mockset, tmp_path):
    es, st = EvalSet(mockset), Stage.parse("visual:separate:test_id")
    with pytest.raises(RuntimeError):
        run_stage(MockVLM("crash_after:5"), es, st, tmp_path, batch_size=2, log=lambda *_: None)
    path = stage_log_path(tmp_path, "mock-crash_after", st)
    assert len(open(path).readlines()) == 4
    with open(path, "a") as fh:
        fh.write('{"key": "half-written')
    model = MockVLM("crash_after:1000")
    res = run_stage(model, es, st, tmp_path, batch_size=4, log=lambda *_: None)
    assert res["run"] == 12 and len(model.seen) == 12
    keys = [json.loads(x)["key"] for x in open(path) if x.strip().endswith("}")]
    assert len(keys) == len(set(keys)) == 16
    assert run_stage(model, es, st, tmp_path, log=lambda *_: None)["run"] == 0


class _ErrModel(MockVLM):
    def generate(self, prompts):
        return [Generation(text="", error="ValueError('prompt too long')") for _ in prompts]


def test_errors_logged_then_retried(mockset, tmp_path):
    es, st = EvalSet(mockset), Stage.parse("described:validation")
    bad = _ErrModel("random")
    bad.tag = "mock-x"
    res = run_stage(bad, es, st, tmp_path, batch_size=4, log=lambda *_: None)
    assert res["stopped"] == "errors" and res["errors"] == 8
    rows, _ = _rows(tmp_path)
    assert rows[0]["n_error"] == 8 and rows[0]["panel_acc"] == 0.0
    good = MockVLM("oracle", answers=_oracle(es))
    good.tag = "mock-x"
    res = run_stage(good, es, st, tmp_path, batch_size=4, log=lambda *_: None)
    assert res["run"] == 16
    rows, _ = _rows(tmp_path)
    assert rows[0]["n_puzzles"] == 16 and rows[0]["n_error"] == 0 and rows[0]["panel_acc"] == 1.0


def test_select_problems_round_robin(mockset):
    es = EvalSet(mockset)
    sel = select_problems(es, "test_id", 8)
    rules = [es.load_problem(p)["rule_id"] for p in sel]
    assert len(sel) == 8 and sel == sorted(sel, key=es.split_ids("test_id").index)
    counts = {r: rules.count(r) for r in set(rules)}
    assert max(counts.values()) - min(counts.values()) <= 1 and len(counts) == 3


def test_adapter_request_shapes(mockset):
    es = EvalSet(mockset)
    prob = es.load_problem("B1-validation-0000")
    pr = build_prompt(prob, "visual", "separate", SPECS["qwen3vl"], es.load_panel)
    content = qwen3vl.to_messages(pr)[0]["content"]
    assert [c["type"] for c in content].count("image") == 14
    assert "image" not in content[1] and content[0]["type"] == "text"
    assert [c["type"] for c in content] == ["text" if k == "text" else "image" for k, _ in pr.parts]
    assert all("problem_id" not in json.dumps(c, default=str) for c in content)
    pr = build_prompt(prob, "visual", "composite", SPECS["internvl"], es.load_panel)
    assert internvl.to_text(pr).count("<image>") == 1
    pr = build_prompt(prob, "symbolic", "none", SPECS["internvl"], es.load_panel)
    assert "<image>" not in internvl.to_text(pr)


def test_budget_estimates(mockset):
    es = EvalSet(mockset)
    est = estimate_stage(es, "qwen3vl", Stage.parse("visual:separate:test_id"))
    assert est["n"] == 16 and est["visual_tokens"] == 3612 and est["images_per_problem"] == 14
    est = estimate_stage(es, "internvl", Stage.parse("visual:composite:test_id"))
    assert est["visual_tokens"] == 3330
    assert estimate_stage(es, "qwen3vl", Stage.parse("described:ood_count"))["missing_split"]


def test_cli_end_to_end(mockset, tmp_path):
    py, script = sys.executable, str(REPO / "scripts" / "run_eval.py")
    out = str(tmp_path / "runs")
    r = subprocess.run([py, script, "plan", "--set", str(mockset), "--out", out, "--dry-run"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "visual:separate:test_id" in r.stdout and "split missing" in r.stdout
    r = subprocess.run([py, script, "run", "--set", str(mockset), "--out", out, "--model", "mock", "--mock-mode",
                        "random", "--stages", "visual:composite:test_id,described:test_id", "--max-problems", "6"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([py, script, "smoke", "--set", str(mockset), "--out", out, "--model", "mock"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([py, script, "score", "--set", str(mockset), "--out", out], capture_output=True, text=True)
    assert r.returncode == 0 and (Path(out) / "summary.csv").exists() and (Path(out) / "b_vs_partner_a.csv").exists()
    r = subprocess.run([py, script, "preview", "--set", str(mockset), "--problem", "B1-test_id-0000",
                        "--dump", str(tmp_path / "pv"), "--spec", "internvl"], capture_output=True, text=True)
    assert r.returncode == 0 and (tmp_path / "pv" / "image_00.png").exists()


def test_plan_executor_with_mock_plan(mockset, tmp_path):
    plan = {"name": "mock", "hours": 1.0, "reserve_hours": 0.0,
            "smoke": {"split": "validation", "max_problems": 3, "stages": ["described"]},
            "stages": [{"priority": 1, "model": "mock", "stage": "visual:separate:test_id"},
                       {"priority": 2, "model": "mock", "stage": "symbolic:ood_render"}]}
    pf = tmp_path / "plan.json"
    pf.write_text(json.dumps(plan))
    out = tmp_path / "runs"
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "run_eval.py"), "plan", "--set", str(mockset),
                        "--out", str(out), "--plan", str(pf), "--model", "mock", "--mock-mode", "left",
                        "--model-dir", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rows, _ = _rows(out)
    assert {(x["condition"], x["split"], x["n_puzzles"]) for x in rows} == {("visual", "test_id", 16), ("symbolic", "ood_render", 16)}
    assert all(x["panel_acc"] == 0.5 for x in rows)
