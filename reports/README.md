# EEGAVR results reports

Behaviour-only evaluation of **Qwen3-VL-8B-Instruct** on controlled Bongard-style puzzles. Each puzzle shows 6 LEFT and 6 RIGHT example panels and 2 test panels (T1, T2) to sort. The rules are Type A (a property of one object), Type B (a relation between objects) or Type C (a count). **Panel accuracy** is the fraction of test panels sorted to the correct side, with chance = 0.50. Each cell has 100 puzzles, so the 95% interval is about ±0.07.

Start with the PDF. It explains every rule, how to read the scores and the main caveats.

## v2.0 evaluation set (13 new rules + 5 A partners; 24,600 model answers; 2026-11-19)
- [EEGAVR_v2_results_report.pdf](v2.0_new13/EEGAVR_v2_results_report.pdf): the full report (5 pages).
- [fig1_rule_by_form_by_split_heatmap.png](v2.0_new13/fig1_rule_by_form_by_split_heatmap.png): panel accuracy for each rule × input form × split.
- [fig2_B_rule_vs_A_partner.png](v2.0_new13/fig2_B_rule_vs_A_partner.png): each relational rule against its property partner on the same base scenes.
- [summary_all_levels.csv](v2.0_new13/summary_all_levels.csv): stage, type and rule-level accuracy, both-correct, Wilson CIs, parse counts and same-side rate. In the rule rows, A2–A5 pool each rule's own puzzles with its partner sets.
- [test_id_per_set.csv](v2.0_new13/test_id_per_set.csv): test_id accuracy per puzzle set (own and partner sets separately).
- [type_by_form_by_split_ownsets.csv](v2.0_new13/type_by_form_by_split_ownsets.csv): Type A/B/C means, own sets only.
- [b_vs_partner_a.csv](v2.0_new13/b_vs_partner_a.csv): paired B-minus-partner differences with CIs.
- [independent_pixel_leak_probe.csv](v2.0_new13/independent_pixel_leak_probe.csv): stimulus control. A simple pixel classifier is at chance on every B and C rule.
- Raw model answers (one JSON line per puzzle, including the model's raw text): [`preliminary_results/v2.0_new13/run_outputs/v2_new13/full/qwen3vl-8b/`](../preliminary_results/v2.0_new13/run_outputs/v2_new13/full/qwen3vl-8b/).

## Pilot v1.1 (6 rules; 11,000 model answers; 2026-11-17)
- [pilot_test_id_heatmap.png](pilot_v1.1/pilot_test_id_heatmap.png), [pilot_summary.csv](pilot_v1.1/pilot_summary.csv), [pilot_by_family.csv](pilot_v1.1/pilot_by_family.csv), [pilot_b_vs_partner_a.csv](pilot_v1.1/pilot_b_vs_partner_a.csv), [pilot_independent_pixel_leak_probe.csv](pilot_v1.1/pilot_independent_pixel_leak_probe.csv).
- Raw model answers: [`preliminary_results/pilot_v1.1/run_outputs/`](../preliminary_results/pilot_v1.1/run_outputs/).

## Provenance
- Code: this repository at commit `70a9c56` (generator 0.4.0, data contract 1.2, harness with prompt `neutral-v2`).
- Evaluation set `eegavr_v2.0_new13`: 6,350 puzzles and 88,900 panels, md5 `689d74e36bf80e3b3ab743f2cf6deb6e`. The datasets are not stored in git (`data/` is ignored; see *Datasets* below).
- Results tarball md5 `8a9e357cc1bddd16a92b0263f0a5988c`. Run on one RTX A6000, 8.39 GPU hours.

## Datasets
The stimulus sets (panel images, scenes, problems) are too large for git. They are not in this repository.
