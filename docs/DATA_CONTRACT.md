# Data contract (v1, 2026-11-16)

Shared by the generator (writer) and the evaluation harness (reader).
Any change bumps `contract_version` in `manifest.json`.

## Coordinates and conventions
- Panel coordinates are fractions of the panel width: x from the left edge, y from the TOP edge, both in [0, 1].
  "Above" therefore means a smaller y. Text descriptions say "from the top" explicitly.
- Object position = area centroid of the filled region (not the bounding-box centre).
- Angles in degrees, counter-clockwise, in [0, 360); orientation comparisons are modulo the shape's symmetry order.
- Radius = circumradius as a fraction of panel width.

## Directory layout (one set = one directory)
```
<set>/
  manifest.json                 # contract_version, generator_version, git commit, rules, splits, counts, model input sizes
  panels/<panel_id>.png         # 512x512, 8-bit grayscale, black ink on white
  masks/<panel_id>.png          # 512x512 uint8 label map: 0 = background, k+1 = object index k (filled region)
  scenes/<panel_id>.json        # scene list (below)
  problems/<problem_id>.json    # problem (below)
  splits/<split>.txt            # one problem_id per line; splits: validation, test_id, ood_count, ood_count_control, ood_render
  audit/report.md               # human-readable audit summary
  audit/<setting_id>.json       # per rule setting: 5a/5b/5c/5d results, probe accuracies with 90% CIs, n pairs, pass/fail
```

## Scene JSON (`scenes/<panel_id>.json`)
```json
{
  "panel_id": "B1-s03-000417",
  "seed": 123456789,
  "generator_version": "0.1.0",
  "render_style": "default",            // "default" or the OOD render style name
  "objects": [
    {"idx": 0, "shape": "triangle", "chiral": false, "symmetry_order": 3,
     "radius": 0.11, "fill": false, "x": 0.31, "y": 0.22, "rotation": 15.0,
     "stroke_px": 4.0, "bbox": [0.20, 0.10, 0.42, 0.30],
     "role": "t1"}                      // "t1", "t2" or "distractor" — HIDDEN from every solver
  ],
  "truth": {"A1": true, "A6": false, "B1": true, "B3": false, "B4": false, "C1": false}
}
```
`truth` holds the value of every pilot rule predicate on this panel (used by audit 5c and probes).

## Problem JSON (`problems/<problem_id>.json`)
```json
{
  "problem_id": "B1-test_id-0001",
  "rule_id": "B1", "type": "B",          // "A", "B", "C"
  "partner_problem_id": "A1p-test_id-0001",   // A partner built from the same base scenes (B rules), else null
  "rule_setting_id": "B1-s03",
  "knob": 0.12,
  "split": "test_id",
  "positive_side": "left",               // side whose panels satisfy the rule
  "left": ["<panel_id>", "... 6 ids, display order"],
  "right": ["<panel_id>", "... 6 ids, display order"],
  "test": [{"panel_id": "<id>", "answer": "left"}, {"panel_id": "<id>", "answer": "right"}],
  "twin_map": {"<positive panel_id>": "<negative panel_id>"},
  "described": {"<panel_id>": "Shape 1: an outline triangle, centre at x=0.31 from the left, y=0.22 from the top, size 0.11, rotated 15 degrees. Shape 2: ..."},
  "symbolic": {"<panel_id>": {"objects": [{"id": 1, "shape": "triangle", "fill": "outline", "x": 0.31, "y": 0.22, "size": 0.11, "rotation": 15.0}],
                               "relations": [["above", 1, 2], ["inside", 3, 1], ["touching", 2, 3]]}}
}
```
- `described` and `symbolic` never contain roles or rule names. Object order in them is random per panel.
- Symbolic relations listed for every ordered pair / triple where true: `above`, `left_of`, `inside`, `touching`, `collinear`, `between` (`["between", m, a, b]` = m lies between a and b), plus `same_shape`.
- The answer for each test panel is the side whose rule value it shares.

## What the harness may read
Everything except `role` (scenes), `truth` (scenes), `twin_map`, `rule_id`, `type`, `rule_setting_id`, `knob`, `partner_problem_id` when building prompts.
Those fields are for scoring and analysis only.
