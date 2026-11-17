import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

from .dataset import select_problems
from .prompts import build_prompt
from .runner import Stage
from .vision import SPECS, image_cost

# Assumed (unmeasured) A6000 throughput; replace with calibrate() numbers from the smoke log.
ASSUMED = {
    "qwen3vl": {"prefill_tps": 2500.0, "decode_tps": 600.0, "out_tokens": 60, "load_s": 420},
    "internvl": {"prefill_tps": 2500.0, "decode_tps": 600.0, "out_tokens": 60, "load_s": 420},
    "mock": {"prefill_tps": 1e9, "decode_tps": 1e9, "out_tokens": 20, "load_s": 0},
}
CHARS_PER_TOKEN = 3.3  # conservative for English/JSON with Qwen-style BPE tokenizers


def _blank(_pid):
    return Image.new("L", (512, 512), 255)


def text_tokens(text, tokenizer=None):
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return int(len(text) / CHARS_PER_TOKEN) + 1


def estimate_stage(evalset, model_key, stage, max_problems=None, sample=12, tokenizer=None):
    spec = SPECS[model_key]
    try:
        ids = select_problems(evalset, stage.split, max_problems)
    except FileNotFoundError:
        return {"n": 0, "missing_split": True}
    if not ids:
        return {"n": 0}
    step = max(1, len(ids) // sample)
    toks, vis, n_img = [], [], []
    for pid in ids[::step][:sample]:
        pr = build_prompt(evalset.load_problem(pid), stage.condition, stage.layout, spec, _blank,
                          load_scene=evalset.load_scene)
        t_text = text_tokens(pr.text_with_placeholders(""), tokenizer) + 30  # + chat-template overhead
        t_vis = sum(image_cost(spec, *im.size)[0] for im in pr.images)
        toks.append(t_text + t_vis)
        vis.append(t_vis)
        n_img.append(len(pr.images))
    k = len(toks)
    return {"n": len(ids), "images_per_problem": sum(n_img) / k, "visual_tokens": sum(vis) / k,
            "prompt_tokens": sum(toks) / k, "max_prompt_tokens": max(toks)}


def seconds_per_problem(model_key, prompt_tokens, calib=None):
    a = ASSUMED[model_key]
    if calib and model_key in calib:
        return prompt_tokens * calib[model_key]["s_per_prompt_token"] + calib[model_key].get("s_fixed", 0.0)
    return prompt_tokens / a["prefill_tps"] + a["out_tokens"] / a["decode_tps"]


def calibrate(out_dir):
    # Effective seconds per prompt token per model family from logged batch timings.
    agg = defaultdict(lambda: [0.0, 0])
    for path in Path(out_dir).rglob("*.jsonl"):
        if path.name == "run_manifest.jsonl":
            continue
        for line in open(path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") != "ok" or not r.get("n_prompt_tokens"):
                continue
            fam = "qwen3vl" if r["model"].startswith("qwen") else "internvl" if r["model"].startswith("internvl") else "mock"
            agg[fam][0] += r["seconds"]
            agg[fam][1] += r["n_prompt_tokens"]
    return {k: {"s_per_prompt_token": s / t, "tokens": t} for k, (s, t) in agg.items() if t}


def plan_table(evalset, plan, hours, max_problems=None, calib=None, tokenizer=None):
    rows, cum, seen_models = [], 0.0, set()
    for st in plan["stages"]:
        model_key, stage = st["model"], Stage.parse(st["stage"])
        if model_key not in seen_models:
            seen_models.add(model_key)
            cum += ASSUMED[model_key]["load_s"]
        mp = st.get("max_problems", max_problems)
        est = estimate_stage(evalset, model_key, stage, mp, tokenizer=tokenizer)
        sec = est["n"] * seconds_per_problem(model_key, est.get("prompt_tokens", 0), calib) if est["n"] else 0.0
        cum += sec
        rows.append({"priority": st.get("priority"), "model": model_key, "stage": str(stage), **est,
                     "est_minutes": sec / 60, "cum_hours": cum / 3600, "fits": cum <= hours * 3600})
    return rows


def format_table(rows):
    hdr = f"{'pri':>3} {'model':9} {'stage':42} {'n':>5} {'img':>4} {'vis_tok':>7} {'tok/prob':>8} {'max_tok':>7} {'min':>7} {'cum_h':>6} fits"
    lines = [hdr]
    for r in rows:
        lines.append(f"{str(r['priority']):>3} {r['model']:9} {r['stage']:42} {r['n']:>5} "
                     f"{r.get('images_per_problem', 0):>4.0f} {r.get('visual_tokens', 0):>7.0f} "
                     f"{r.get('prompt_tokens', 0):>8.0f} {r.get('max_prompt_tokens', 0):>7.0f} "
                     f"{r['est_minutes']:>7.1f} {r['cum_hours']:>6.2f} {'yes' if r['fits'] else 'NO'}"
                     + ("  (split missing)" if r.get("missing_split") else ""))
    return "\n".join(lines)
