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

## Changes in contract 1.1 (2026-11-16, generator track; all additive)
`manifest.json` carries `"contract_version": "1.1"`. A v1 reader that ignores unknown keys can read a 1.1 set unchanged,
except for an explicit version check (the eval harness accepts only `1`; add `"1.1"` to its supported versions).
- Scene objects: `size` = perimeter-equivalent radius (outline perimeter = 2*pi*size; sampled in 0.06–0.16).
  `radius` keeps its v1 meaning (circumradius about the area centroid). `mirror` (bool) = reflected before rotation;
  only meaningful for chiral shapes. `symmetry_order` is 0 for the circle (orientation undefined). Shape names:
  `circle, square, triangle, pentagon, cross, L, right_trapezoid, irregular_pentagon`.
- Scene: `regen` = what `eegavr.generator.assemble.regenerate()` needs besides `seed`
  (`setting_id`, `split`, `counter`, `control`; for A-partner panels also `base_setting_id`, `base_seed`, `salt`).
  Scoring/regeneration only: never show it to a solver.
- `seed` of a panel is the seed of its twin pair (both panels of a pair share it), so seeds reveal pairing; never show them.
- Problems: unchanged keys. `described` gives x, y and size (= `radius`) to 3 decimals, rotation in whole degrees
  (omitted for circles) and "mirrored" for reflected chiral shapes. `symbolic` objects add `mirrored` for chiral shapes.
  Symmetric relations (`touching`, `collinear`, `same_shape`) are listed once with ascending ids; `between` = `[m, a, b]`
  with a < b; `collinear` = every centroid within 0.01 of the line through the other two.
- Scoring-only problem fields also include `positive_side` and `test[*].answer`.
- `audit/` also holds `audit_runs_log.jsonl` (every audit run, all generator versions) and `rendering_check.json`
  (per-panel 5d on the exported set, PIL BOX, 224/256/448 px).
- `manifest.json` adds `a_partner_settings`, `tuning`, `render_styles`, `not_applicable`, `audit_status`,
  `dropped`, `assembly_stats`.

## Changes in contract 1.2 (2026-11-17, integration track; full set v2.0)
`manifest.json` carries `"contract_version": "1.2"`. Readers of 1.1 read 1.2 unchanged apart from the version check
(the harness accepts `"1.2"`).
- Rules: the 19-rule library (A1–A6, B1–B9, C1b–C3b, C4b; C1 and C4 are excluded) and A partners
  `A2pB2, A3pB5, A4pB6, A5pB7, A2pB8, A3pB9` besides the pilot's `A1pB1, A6pB3, A1pB4`.
  `manifest.rules[r].a_partner` names the partner of every B rule.
- Scene `truth` holds every library predicate (additive keys `A3 A4 A5 B2 B5 B6 B7 B8 B9 C2b C3b C4 C4b`).
- Symbolic relations (`symbolic[*].relations`), recomputed for every v2.0 problem, frozen pilot problems included:
  - `collinear` = `[a, b, c]` when the height of the centroid triangle over its longest side is <= 0.004 (the B5/B8
    tolerance; was: every centroid within 0.01 of the line through the other two). `between` = `[m, a, b]` for each
    collinear triple, m = the object opposite the longest side.
  - new `same_orientation` = `[a, b]`: same non-circular shape and handedness, orientations equal modulo the shape's
    symmetry within 1 degree (B6).
  - new `mirror_image` = `[a, b]`: same chiral shape, opposite `mirrored` flag (B7).
  - new `closest_pair` = `[a, b]`: the pair with the smallest centroid distance, one per panel (B9).
  - With `above`, `left_of`, `inside`, `touching` and `same_shape`, every B rule is a function of the relation list
    and the object shapes (tested in `tests/test_relations_v12.py`), so `symbolic` is an upper bound for all nine.
- `symbolic_complete` is built by the harness from `scenes/` at prompt time (`eegavr/eval/geometry.py`); it is
  unchanged by 1.2 and therefore consistent with v2.0 by construction.
- Frozen pilot problems (copied from pilot v1.1): panel images, masks, scene objects and problem composition are
  unchanged; `described`/`symbolic` are recomputed from the exactly regenerated scenes and `truth` gains the new keys.
- `manifest.json` adds `seed_code_blocks`; `assembly_stats` adds `partner_restarts` and `partner_infeasible_salts`.
- Seeds: v2.0 settings use fixed code blocks (`settings.CODE_BLOCK`), disjoint from the pilot codes 1–40, so no v2.0
  seed equals a v1.1 seed.
