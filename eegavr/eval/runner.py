import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .dataset import select_problems
from .parse import parse_answer
from .prompts import PROMPT_VERSION, build_prompt
from .vision import SPECS


@dataclass(frozen=True)
class Stage:
    condition: str
    layout: str
    split: str

    @classmethod
    def parse(cls, s):
        # "visual:separate:test_id" or "described:test_id"
        bits = s.split(":")
        if len(bits) == 2:
            return cls(bits[0], "none", bits[1])
        return cls(bits[0], bits[1] if bits[0] == "visual" else "none", bits[2])

    def __str__(self):
        return f"{self.condition}:{self.layout}:{self.split}"


def git_commit(path=None):
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=path or Path(__file__).parent,
                              capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def record_key(model_tag, condition, layout, problem_id):
    return f"{model_tag}|{condition}|{layout}|{problem_id}|{PROMPT_VERSION}"


def stage_log_path(out_dir, model_tag, stage):
    return Path(out_dir) / model_tag / f"{stage.condition}-{stage.layout}" / f"{stage.split}.jsonl"


def read_done(path, retry_errors=True):
    # Keys already logged; a truncated final line (crash mid-write) is ignored.
    done = set()
    if not Path(path).exists():
        return done
    with open(path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if retry_errors and r.get("status") == "error":
                done.discard(r["key"])
            else:
                done.add(r["key"])
    return done


def _append(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        with open(path, "rb") as fh:
            fh.seek(-1, 2)
            torn = fh.read(1) != b"\n"
    else:
        torn = False
    with open(path, "a") as fh:
        if torn:
            fh.write("\n")  # a crash left a partial line; start a fresh one
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def run_stage(model, evalset, stage, out_dir, batch_size=16, max_problems=None, deadline=None,
              retry_errors=True, log=print, max_error_batches=2):
    # Runs one (condition, layout, split) for a loaded model; resumable; returns counts.
    spec = SPECS[model.spec_key]
    path = stage_log_path(out_dir, model.tag, stage)
    done = read_done(path, retry_errors)
    ids = select_problems(evalset, stage.split, max_problems)
    todo = [pid for pid in ids if record_key(model.tag, stage.condition, stage.layout, pid) not in done]
    log(f"[{model.tag} {stage}] {len(ids)} selected, {len(ids) - len(todo)} done, {len(todo)} to run")
    n_run = n_err = err_batches = 0
    set_digest, commit, t_stage = evalset.digest(), git_commit(), time.time()
    sec_per_batch = None
    for b0 in range(0, len(todo), batch_size):
        if deadline and sec_per_batch and time.time() + sec_per_batch > deadline:
            log(f"[{model.tag} {stage}] deadline reached; stopping with {len(todo) - b0} left")
            return {"stage": str(stage), "run": n_run, "errors": n_err, "left": len(todo) - b0, "stopped": "deadline"}
        batch = todo[b0:b0 + batch_size]
        problems = [evalset.load_problem(pid) for pid in batch]
        prompts = [build_prompt(p, stage.condition, stage.layout, spec, evalset.load_panel, evalset.load_scene)
                   for p in problems]
        t0 = time.time()
        gens = model.generate(prompts)
        dt = time.time() - t0
        sec_per_batch = dt if sec_per_batch is None else 0.7 * sec_per_batch + 0.3 * dt
        recs = []
        for prob, prompt, g in zip(problems, prompts, gens):
            parsed = parse_answer(g.text) if not g.error else {"T1": None, "T2": None, "status": "error"}
            truth = {"T1": prob["test"][0]["answer"], "T2": prob["test"][1]["answer"]}
            recs.append({
                "key": record_key(model.tag, stage.condition, stage.layout, prob["problem_id"]),
                "model": model.tag, "condition": stage.condition, "layout": stage.layout, "split": stage.split,
                "problem_id": prob["problem_id"], "rule_id": prob.get("rule_id"), "type": prob.get("type"),
                "partner_problem_id": prob.get("partner_problem_id"), "rule_setting_id": prob.get("rule_setting_id"),
                "knob": prob.get("knob"), "truth": truth, "raw_output": g.text,
                "status": "error" if g.error else "ok", "error": g.error, "parse_status": parsed["status"],
                "t1_correct": parsed["T1"] == truth["T1"], "t2_correct": parsed["T2"] == truth["T2"],
                "n_images": len(prompt.images), "image_sizes": prompt.meta["image_sizes"],
                "n_prompt_tokens": g.n_prompt_tokens, "n_output_tokens": g.n_output_tokens,
                "finish_reason": g.finish_reason, "seconds": dt / len(batch), "batch_seconds": dt,
                "batch_size": len(batch), "prompt_version": PROMPT_VERSION, "set_digest": set_digest,
                "harness_commit": commit, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
        _append(path, recs)
        n_run += len(recs)
        e = sum(r["status"] == "error" for r in recs)
        n_err += e
        err_batches = err_batches + 1 if e == len(recs) else 0
        if err_batches >= max_error_batches:
            log(f"[{model.tag} {stage}] {err_batches} consecutive all-error batches; aborting stage. Last: {recs[-1]['error']}")
            return {"stage": str(stage), "run": n_run, "errors": n_err, "left": len(todo) - b0 - len(batch),
                    "stopped": "errors"}
        log(f"[{model.tag} {stage}] {b0 + len(batch)}/{len(todo)}  {dt / len(batch):.2f} s/problem  errors={n_err}")
    log(f"[{model.tag} {stage}] finished in {time.time() - t_stage:.0f} s")
    return {"stage": str(stage), "run": n_run, "errors": n_err, "left": 0, "stopped": None}


def write_run_manifest(out_dir, model, evalset, extra=None):
    path = Path(out_dir) / model.tag / "run_manifest.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": model.info(), "set_root": str(evalset.root),
           "set_digest": evalset.digest(), "set_manifest": evalset.manifest, "prompt_version": PROMPT_VERSION,
           "harness_commit": git_commit(), **(extra or {})}
    with open(path, "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
