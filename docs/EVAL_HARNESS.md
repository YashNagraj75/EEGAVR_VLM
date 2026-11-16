# Evaluation harness (eegavr/eval, scripts/run_eval.py)

Reads any set in the `docs/DATA_CONTRACT.md` format (contract v1) and evaluates open VLMs zero-shot.
Run instructions: `docs/GPU_RUNBOOK.md`.

## Modules
- `dataset.py`: set reader. `prompt_view()` is the only path from a problem file to a prompt. It keeps `left`, `right`, the test panel ids, `described` and `symbolic`. It drops every scoring-only field (`rule_id`, `type`, `rule_setting_id`, `knob`, `partner_problem_id`, `twin_map`, `positive_side`, `split`, `test[*].answer`). It raises `LeakageError` if `described`/`symbolic` contain `role`, `truth`, `answer`, `panel_id`, ... keys, or any panel/problem id.
- `vision.py`: per-model image geometry, the composite renderer, and copies of the Qwen `smart_resize` and InternVL tiling rules (vLLM 0.33.0 / transformers 5.23).
- `prompts.py`: the prompt builder (`PROMPT_VERSION = "neutral-v1"`).
- `parse.py`: answer parser. `scoring.py`: metrics and CSVs. `runner.py`: resumable stage runner. `budget.py`: token/time estimates.
- `models/`: `qwen3vl.py`, `internvl.py` (vLLM engine + transformers fallback each), `mock.py`. Model code is isolated there, and adapters read only `Prompt.parts`.
- `mockset.py`: a tiny contract-v1 fixture set for the tests (not real stimuli).

## Input conditions and layouts (protocol Step 9)
All prompts share one neutral instruction (both groups are named; the rule may hold on either side; nothing about rule
kinds), then the panels, then the answer format `{"T1": "left|right", "T2": "left|right", "rule": "<one sentence>"}`.
Panels are always labelled L1-L6, R1-R6, T1, T2 in file display order. Nothing is randomised, and panel/problem ids never appear in a prompt.

| Condition | Layout | Content |
|---|---|---|
| visual | separate | 14 images, each preceded by its text label, under "LEFT group", "RIGHT group" and "Test panels" headers |
| visual | composite | one image: LEFT 2x3 block (top left), RIGHT 2x3 block (top right), a gap, T1 and T2 centred below; every panel labelled and framed |
| described | - | the problem's `described` text per panel, plus one sentence on the coordinate convention |
| symbolic | - | the problem's `symbolic` JSON per panel, plus a legend of the object fields and the relation notation |

## Image resolutions
The harness resizes (area averaging, PIL `BOX`) and pads itself, so the model processor's own resize is a no-op. Qwen
images have both sides divisible by 32 and lie inside the default pixel range [65,536, 16,777,216]. InternVL images are exact
448-px tile grids, so the closest-aspect rule picks 1x1 for a panel and 4x3 (+ thumbnail) for the composite.

| Model | Layout | Image(s) | Effective panel | Visual tokens |
|---|---|---|---|---|
| Qwen3-VL-8B-Instruct | separate | 14 x 512x512 | 512 px | 14 x 258 = 3,612 |
| Qwen3-VL-8B-Instruct | composite | 2656x1632 | 384 px | 4,235 |
| InternVL3.5-8B | separate | 14 x 448x448 | 448 px | 14 x 258 = 3,612 |
| InternVL3.5-8B | composite | 1792x1344 (12 tiles + thumbnail) | 256 px | 3,330 |

**Smallest effective panel size: 256 px** (InternVL3.5 composite). The generator's 5d check should use 256 px with
BOX downsampling: a rule-relevant difference of 2 px there is 0.0078 of the panel width (about 4 px on the 512 master).
The composite panel sizes keep the visual-token budget of the two layouts roughly equal within each model, but composite
panels have fewer pixels, so layout and resolution are partly confounded.

## Decoding
Greedy (`temperature=0`), `max_tokens=256`, no system prompt (model default template), no thinking mode, no guided/JSON-constrained decoding. Constrained decoding would remove format failures but would change the model's behaviour, so it was not used.
vLLM: bf16, `max_model_len=32768`, `limit_mm_per_prompt={"image": 14}`, `max_num_seqs` = batch size (16), seed 0.

## Parsing
`<think>` blocks and code fences are stripped. The parser tries every `{...}` object from the last to the first, first strictly (`json`) and then after
repairs for single quotes, unquoted keys and trailing commas (`json_repaired`). If that fails, a label-anchored regex is tried (`regex`). Side values are normalised: `"Left"`, `"left group"` and `"L"` all become left. A value that contains both sides (e.g. an echoed `"left|right"`) is invalid.
Statuses: `json`, `json_repaired`, `regex` (scored), `partial` (only one panel answered; the missing one is wrong),
`unparsable` (both wrong), `error` (engine error; both wrong, retried on resume). Every status is counted in `summary.csv`, together with the
strict-JSON rate.

## Scoring
- Per-panel accuracy over the 2 test panels per problem (chance 0.5); per-puzzle both-correct (chance 0.25).
- Wilson 95% CIs; exact one-sided binomial p against chance.
- Levels: per stage (model x condition x layout x split), per type (A/B/C), per rule.
- Diagnostics: share of "left" answers, and the share of puzzles with the same side for T1 and T2 (the true answers always differ).
- B vs partner A: each B problem is paired with its `partner_problem_id` (same model, condition and layout). The output gives the mean
  per-panel and both-correct difference B - A, with a 2,000-resample bootstrap 95% CI over pairs and McNemar discordant counts.
- Scoring re-parses `raw_output`, so the results always reflect the current parser.

## Not verified without a GPU (huggingface.co is unreachable from the sandbox)
- Loading either model, the chat templates (`AutoProcessor.apply_chat_template` for Qwen, `AutoTokenizer` + `<image>` for InternVL), and vLLM multi-image requests.
- That the processors really leave the images unchanged (the smoke-test token check verifies this).
- Throughput (the plan uses ASSUMED 2,500 prompt tokens/s until it is calibrated).
- The transformers fallbacks (`Qwen3VLForConditionalGeneration`, `AutoModelForImageTextToText` on `OpenGVLab/InternVL3_5-8B-HF`).
- InternVL3.5's default chat template and stop tokens under vLLM 0.33.
