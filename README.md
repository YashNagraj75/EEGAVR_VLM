# EEGAVR_VLM

Controlled Bongard-style stimuli and VLM evaluation for the EEGAVR project
(does a vision-language model compute spatial relations?).

- `eegavr/generator/` — scene lists, rules, near-miss twins, rendering, problem assembly, exports
- `eegavr/audit/` — shortcut audits 5a–5d (pixel, cross-type leakage, competing rules, rendering)
- `eegavr/eval/` — VLM evaluation harness (Qwen3-VL-8B-Instruct, InternVL3.5-8B)
- `configs/` — rule settings, split sizes, model configs
- `docs/DATA_CONTRACT.md` — the on-disk format shared by the generator and the harness
- `docs/EVAL_HARNESS.md` — harness design (prompts, resolutions, parsing, scoring); `docs/GPU_RUNBOOK.md` — A6000 run instructions
- `scripts/` — command-line entry points

The design follows the "Controlled stimulus curation protocol" (EEGAVR wiki).
The stimuli are controlled Methods stimuli, not a dataset contribution.
