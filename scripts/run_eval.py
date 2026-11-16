import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eegavr.eval.budget import calibrate, format_table, plan_table, text_tokens
from eegavr.eval.dataset import EvalSet
from eegavr.eval.prompts import build_prompt
from eegavr.eval.runner import Stage, run_stage, write_run_manifest
from eegavr.eval.scoring import score_dir
from eegavr.eval.vision import SPECS, image_cost, layout_summary


def _model(args):
    from eegavr.eval.models import get_model
    kw = {} if args.model == "mock" else dict(max_tokens=args.max_tokens, max_model_len=args.max_model_len,
                                              gpu_memory_utilization=args.gpu_mem, max_num_seqs=args.batch_size)
    if args.model == "mock":
        kw = {"mode": args.mock_mode}
    return get_model(args.model, backend=args.backend, model_path=args.model_path, **kw)


def _tokenizer(model):
    tok = getattr(model, "tokenizer", None) or getattr(getattr(model, "processor", None), "tokenizer", None)
    return tok


def token_check(model, evalset, stages, out, n=3):
    # Prints logged minus predicted prompt tokens; a constant residual confirms the image sizes.
    tok = _tokenizer(model)
    spec = SPECS[model.spec_key]
    from eegavr.eval.runner import stage_log_path
    for st in stages:
        path = stage_log_path(out, model.tag, st)
        if not path.exists():
            continue
        recs = [json.loads(line) for line in open(path)][:n]
        res = []
        for r in recs:
            if not r.get("n_prompt_tokens"):
                continue
            pr = build_prompt(evalset.load_problem(r["problem_id"]), st.condition, st.layout, spec, evalset.load_panel)
            vis = sum(image_cost(spec, *im.size)[0] for im in pr.images)
            res.append(r["n_prompt_tokens"] - text_tokens(pr.text_with_placeholders(""), tok) - vis)
        print(f"[token-check {st}] residual tokens (should be a small constant across stages): {res}")


def cmd_run(args, stages=None, deadline=None):
    evalset = EvalSet(args.set)
    stages = stages or [Stage.parse(s) for s in args.stages.split(",")]
    model = _model(args)
    write_run_manifest(args.out, model, evalset, {"argv": sys.argv})
    results = []
    for st in stages:
        res = run_stage(model, evalset, st, args.out, batch_size=args.batch_size, max_problems=args.max_problems,
                        deadline=deadline or args.deadline_epoch, retry_errors=not args.no_retry_errors)
        results.append(res)
        if res["stopped"] == "deadline":
            break
    if args.token_check:
        token_check(model, evalset, stages, args.out)
    print(json.dumps(results, indent=1))
    return results


def cmd_smoke(args):
    plan = json.loads(Path(args.plan).read_text())
    sm = plan["smoke"]
    args.max_problems = args.max_problems or sm["max_problems"]
    stages = [Stage.parse(f"{s}:{sm['split']}") for s in sm["stages"]]
    args.token_check = True
    t0 = time.time()
    cmd_run(args, stages)
    print(f"smoke wall time {time.time() - t0:.0f} s; calibration: {json.dumps(calibrate(args.out))}")
    records, summary, _ = score_dir(args.out)
    for row in summary:
        if row["level"] == "stage":
            print({k: row[k] for k in ("model", "condition", "layout", "split", "n_puzzles", "panel_acc",
                                       "n_json", "n_json_repaired", "n_regex", "n_partial", "n_unparsable", "n_error")})


def cmd_plan(args):
    plan = json.loads(Path(args.plan).read_text())
    hours = args.hours or plan["hours"]
    evalset = EvalSet(args.set)
    calib = calibrate(args.calibrate_from) if args.calibrate_from else None
    rows = plan_table(evalset, plan, hours - plan.get("reserve_hours", 0), args.max_problems, calib)
    print(format_table(rows))
    print("calibration:", json.dumps(calib) if calib else "none (ASSUMED throughput, see eegavr/eval/budget.py)")
    for key in ("qwen3vl", "internvl"):
        print(key, json.dumps(layout_summary(SPECS[key])))
    if args.dry_run:
        return
    deadline = time.time() + (hours - plan.get("reserve_hours", 0)) * 3600
    groups = []
    for st in plan["stages"]:
        if groups and groups[-1][0] == st["model"]:
            groups[-1][1].append(st["stage"])
        else:
            groups.append((st["model"], [st["stage"]]))
    for model_key, stages in groups:
        if time.time() > deadline - 600:
            print(f"skipping {model_key}: <10 min left")
            continue
        cmd = [sys.executable, __file__, "run", "--set", args.set, "--out", args.out, "--model", model_key,
               "--backend", args.backend, "--stages", ",".join(stages), "--deadline-epoch", str(deadline),
               "--batch-size", str(args.batch_size), "--max-tokens", str(args.max_tokens),
               "--max-model-len", str(args.max_model_len), "--gpu-mem", str(args.gpu_mem)]
        if args.max_problems:
            cmd += ["--max-problems", str(args.max_problems)]
        if model_key == "mock":
            cmd += ["--mock-mode", args.mock_mode]
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=False, env=os.environ.copy())
    score_dir(args.out)
    print(f"scored -> {Path(args.out) / 'summary.csv'}")


def cmd_score(args):
    records, summary, pairs = score_dir(args.out)
    print(f"{len(records)} records; summary.csv and b_vs_partner_a.csv written to {args.out}")
    for row in summary:
        if row["level"] in ("stage", "type"):
            print(f"{row['model']:15} {row['condition']:9} {row['layout']:9} {row['split']:18} {str(row.get('type', '')):3} "
                  f"n={row['n_puzzles']:4} panel={row['panel_acc']:.3f} [{row['panel_ci_lo']:.3f},{row['panel_ci_hi']:.3f}] "
                  f"both={row['both_acc']:.3f} unparsable={row['n_unparsable']} errors={row['n_error']}")


def cmd_preview(args):
    from eegavr.eval.prompts import build_prompt as bp
    evalset = EvalSet(args.set)
    spec = SPECS[args.spec]
    st = Stage.parse(args.stage)
    pr = bp(evalset.load_problem(args.problem), st.condition, st.layout, spec, evalset.load_panel)
    out = Path(args.dump)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompt.txt").write_text(pr.text_with_placeholders("<image>"))
    for k, im in enumerate(pr.images):
        im.save(out / f"image_{k:02d}.png")
    print(f"wrote {out}/prompt.txt and {len(pr.images)} images; sizes {pr.meta['image_sizes']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="EEGAVR VLM evaluation harness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--set", required=True, help="data-contract set directory")
    common.add_argument("--out", default="runs/pilot")
    common.add_argument("--model", default="qwen3vl", choices=["qwen3vl", "internvl", "mock"])
    common.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    common.add_argument("--model-path", default=None, help="local weights dir (default: HF repo id)")
    common.add_argument("--max-problems", type=int, default=None, help="cap per stage (round-robin over rules)")
    common.add_argument("--batch-size", type=int, default=16)
    common.add_argument("--max-tokens", type=int, default=256)
    common.add_argument("--max-model-len", type=int, default=16384)
    common.add_argument("--gpu-mem", type=float, default=0.90)
    common.add_argument("--mock-mode", default="random")
    common.add_argument("--no-retry-errors", action="store_true")
    common.add_argument("--plan", default=str(Path(__file__).resolve().parents[1] / "configs" / "eval_plan_a6000_9h.json"))
    common.add_argument("--deadline-epoch", type=float, default=None)
    common.add_argument("--token-check", action="store_true")
    p = sub.add_parser("run", parents=[common])
    p.add_argument("--stages", required=True, help="comma list, e.g. visual:separate:test_id,described:test_id")
    sub.add_parser("smoke", parents=[common])
    p = sub.add_parser("plan", parents=[common])
    p.add_argument("--hours", type=float, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--calibrate-from", default=None, help="run dir with smoke logs, to replace ASSUMED throughput")
    sub.add_parser("score", parents=[common])
    p = sub.add_parser("preview", parents=[common])
    p.add_argument("--problem", required=True)
    p.add_argument("--stage", default="visual:composite:test_id")
    p.add_argument("--spec", default="qwen3vl", choices=sorted(SPECS))
    p.add_argument("--dump", default="runs/preview")
    args = ap.parse_args(argv)
    {"run": cmd_run, "smoke": cmd_smoke, "plan": cmd_plan, "score": cmd_score,
     "preview": cmd_preview}[args.cmd](args)


if __name__ == "__main__":
    main()
