# GPU runbook: VLM evaluation on one A6000 (48 GB), 9-hour window

Status: written and tested only against the mock model on a CPU sandbox (2026-11-16). Every step
that loads a real model is **unverified** until it has run on the GPU host; the smoke test in
step 5 is the verification. Record what you observe in the wiki section "GPU run findings".

## 0. Time plan for the 9 hours

| Step | Expected wall time |
|---|---|
| 1-2 Environment + repo + data copy | 15-25 min |
| 3 Model downloads (~17 GB each, both in parallel) | 10-30 min (bandwidth-bound) |
| 4 Unit tests + dry-run | 2 min |
| 5 Smoke test, both models (10 validation problems x 5 condition/layouts) | ~20 min (mostly engine start-up) |
| 6 Full plan (priority order, stops itself at the deadline) | remaining time, minus 45 min reserve |
| 7 Score + copy results out | 5 min |

Start the downloads (step 3) as soon as the environment exists. Run the full plan inside `tmux`.

## 1. Environment (fresh Linux host)

```bash
nvidia-smi                                   # expect RTX A6000, 48 GB; note the driver version
python3 --version                            # 3.10-3.12
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
uv venv -p 3.12 ~/venv-eval && source ~/venv-eval/bin/activate
uv pip install "vllm==0.33.0"                # pulls torch 2.14.0 and transformers >=5.16.1,<5.21
uv pip install "huggingface_hub[cli]" pillow numpy pandas pytest
python -c "import torch, vllm, transformers; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), vllm.__version__, transformers.__version__)"
```

- vLLM 0.33.0 was the latest release on 2026-11-16. Its pins (torch 2.14.0, transformers 5.16-5.20) were read from the wheel metadata. Qwen3-VL needs vLLM >= 0.11 and transformers >= 4.57, so both are met.
- If `torch.cuda.is_available()` is False, the CUDA build of the torch wheel needs a newer driver than the host has. Either install a vLLM release whose torch matches the driver (`uv pip install "vllm==<older>"`), or use the transformers fallback: `uv pip install torch "transformers>=4.57" accelerate` and pass `--backend hf` (batch size 1, roughly 3-5x slower; see section 6).
- InternVL3.5 under vLLM needs `trust_remote_code` (the adapter sets it). No flash-attn install is needed.
- Thread limits for the CPU-side image work: `export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8`.

## 2. Repository and data

```bash
git clone https://github.com/YashNagraj75/EEGAVR_VLM ~/EEGAVR_VLM && cd ~/EEGAVR_VLM
# if the harness commits are not pushed yet: copy the patch series tarball and apply it
#   tar xzf EEGAVR_VLM_eval_patches.tar.gz && git am patches/*.patch
uv pip install -e .
git log --oneline | head -3                  # record the commit in the wiki
```

Copy a data set in (from the machine that holds it):

```bash
rsync -a --info=progress2 <local>/pilot_set/ <user>@<gpu-host>:~/EEGAVR_VLM/data/pilot/
# or: tar czf pilot_set.tgz -C <local> pilot_set && scp pilot_set.tgz <host>: && tar xzf pilot_set.tgz -C ~/EEGAVR_VLM/data/ && mv ~/EEGAVR_VLM/data/pilot_set ~/EEGAVR_VLM/data/pilot
md5sum data/pilot/manifest.json              # compare with the sender's md5
```

The set must follow `docs/DATA_CONTRACT.md` (contract v1). `data/` and `runs/` are git-ignored.

## 3. Model downloads

```bash
mkdir -p ~/models
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir ~/models/Qwen3-VL-8B-Instruct &
hf download OpenGVLab/InternVL3_5-8B     --local-dir ~/models/InternVL3_5-8B &
wait
# only if you need the transformers fallback for InternVL:
#   hf download OpenGVLab/InternVL3_5-8B-HF --local-dir ~/models/InternVL3_5-8B-HF
```

`--model-dir ~/models` resolves each model to `~/models/<repo name>`. The commit sha of the weights
is written to `runs/<out>/<model>/run_manifest.jsonl`. It comes from the download metadata, or from the Hub API if the metadata is missing.

## 4. Unit tests and dry-run (CPU, ~1 min)

```bash
python -m pytest -q tests                    # expect: 81 passed (or more)
python scripts/run_eval.py plan --set data/pilot --dry-run
```

The dry-run prints, for every stage in `configs/eval_plan_a6000_9h.json`: number of problems,
images per problem, visual tokens, mean and max prompt tokens, estimated minutes and the cumulative hours. The
throughput is an ASSUMPTION (2,500 prompt tokens/s effective, see `eegavr/eval/budget.py`) until step 5
replaces it. Any `WARNING ... exceeds --max-model-len` line means you must raise `--max-model-len`
(default 32768) before the full run.

## 5. Smoke test (10 validation problems, all five condition/layouts, per model)

```bash
python scripts/run_eval.py smoke --set data/pilot --out runs/smoke --model qwen3vl  --model-dir ~/models
python scripts/run_eval.py smoke --set data/pilot --out runs/smoke --model internvl --model-dir ~/models
python scripts/run_eval.py plan  --set data/pilot --dry-run --calibrate-from runs/smoke
```

Pass criteria (write the numbers into the wiki):
1. No `error` records and no engine crash; `n_unparsable` small (a few of 10 is a prompt problem, not a model result: stop and report).
2. `[token-check ...] residual tokens` is the same small constant (the chat-template overhead, roughly 5-30) for the visual and the text stages. A visual-stage residual that differs by hundreds or thousands of tokens means the model processor resized the images: the effective panel sizes in `docs/EVAL_HARNESS.md` are then wrong, and the run must not continue until this is fixed.
3. Check `n_prompt_tokens` in `runs/smoke/qwen3vl-8b/visual-separate/validation.jsonl`. Expect about 3,612 visual tokens + ~300 text tokens. For the composite expect about 4,235 visual tokens (Qwen) and about 3,330 (InternVL).
4. Look at a few `raw_output` strings by eye.
5. The calibrated dry-run shows the real minutes per stage; note which stages fit.

Validation problems are used only here. The test sets are untouched until the full run.

## 6. Full run (priority order, stops at the deadline)

```bash
tmux new -s eval
cd ~/EEGAVR_VLM && source ~/venv-eval/bin/activate
python scripts/run_eval.py plan --set data/pilot --out runs/pilot --model-dir ~/models --hours <hours left> 2>&1 | tee runs/pilot_plan.log
```

- `--hours` is the time still left in the window. A 0.75 h reserve is subtracted for scoring and copying.
  The runner does not start a batch that would end after the deadline. Each model runs in its own process (one vLLM engine per process).
- Resumable: if anything dies, rerun the same command; finished problems are skipped (key = model,
  condition, layout, problem, prompt version); error records are retried.
- Priority order (`configs/eval_plan_a6000_9h.json`, pilot first: pilot v1.1 in `data/pilot`): Qwen visual-separate
  test_id -> described -> symbolic -> symbolic_complete -> visual-composite -> Qwen OOD (visual separate on ood_count,
  ood_count_control, ood_render; then described, symbolic and symbolic_complete on the two count splits; then composite
  OOD) -> the same 17 stages for InternVL3.5. The text conditions are not run on ood_render, because the scene text does
  not change with the render style (symbolic_complete would differ only in `line_width`).
- To cap a stage (e.g. if the calibrated plan does not fit): `--max-problems N` takes N per stage, round-robin over rules in
  split-file order, so every rule keeps the same weight. Report any cap as a deviation.
- Expected time per stage (ASSUMED throughput, mock-set prompt lengths; replace with the calibrated table):

| Condition / layout | Prompt tokens per problem | s per problem | min per 100 problems |
|---|---|---|---|
| visual separate (both models) | ~3,900 | ~1.7 | ~2.8 |
| visual composite, Qwen / InternVL | ~4,540 / ~3,630 | ~1.9 / ~1.6 | ~3.2 / ~2.6 |
| described | ~1,600 (more with 7-8 objects) | ~0.7 | ~1.2 |
| symbolic | ~2,000 (much more with 7-8 objects) | ~0.9 | ~1.5 |
| symbolic_complete | ~2,600 test_id, ~4,700 ood_count (pilot v1.1) | ~1.1 / ~2.0 | ~1.9 / ~3.3 |

  Add ~7 min of engine start-up per model. Dry-run on pilot v1.1 (2026-11-17, ASSUMED throughput): the 17 Qwen stages end at
  5.1 h, and InternVL3.5 fits up to priority 27 (described:ood_count_control, 8.1 h) of the 8.25 h usable. The largest
  prompt is about 8,100 tokens (symbolic, ood_count), well inside `--max-model-len 32768`. The calibrated dry-run decides
  what really fits.
- The transformers fallback (`--backend hf`) runs one problem at a time. Expect ~5-10 s per visual problem, so only the first
  three or four Qwen stages fit.
- Monitor: `watch -n 30 nvidia-smi`; `tail -f runs/pilot_plan.log`; `wc -l runs/pilot/*/*/*.jsonl`.

## 7. Score and copy results out

```bash
python scripts/run_eval.py score --set data/pilot --out runs/pilot
tar czf eval_results_$(date +%Y%m%d_%H%M).tgz runs/pilot runs/smoke runs/pilot_plan.log
# from your machine:
scp <user>@<gpu-host>:~/EEGAVR_VLM/eval_results_*.tgz .
```

Outputs: `runs/pilot/<model>/<condition>-<layout>/<split>.jsonl` (one record per problem: raw output,
parsed answer, parse status, correctness, prompt/output tokens, seconds, image sizes, set digest, harness
commit), `runs/pilot/<model>/run_manifest.jsonl` (model revision, library versions, engine args),
`runs/pilot/summary.csv` (per stage, per type, per rule: per-panel accuracy with Wilson 95% CI vs chance 0.5,
both-correct with CI vs chance 0.25, exact one-sided binomial p, parse-status counts, left-bias, same-side rate,
tokens, time) and `runs/pilot/b_vs_partner_a.csv` (paired B minus A-partner differences with bootstrap 95% CIs).
Scoring re-parses the raw outputs, so a later parser fix can be applied to old logs.
