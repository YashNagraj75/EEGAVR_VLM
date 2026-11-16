import importlib
import json
from pathlib import Path

from .base import Generation


def versions():
    out = {}
    for mod in ("torch", "vllm", "transformers", "PIL"):
        try:
            out[mod] = importlib.import_module(mod).__version__
        except Exception:
            out[mod] = None
    return out


def model_revision(path, repo_id):
    # Commit sha of the weights: local `hf download` metadata first, then the Hub API.
    meta = Path(path) / ".cache" / "huggingface" / "download"
    if meta.exists():
        for f in meta.rglob("config.json.metadata"):
            lines = f.read_text().splitlines()
            if lines:
                return lines[0].strip()
    try:
        from huggingface_hub import model_info
        return model_info(repo_id).sha
    except Exception:
        return None


def vllm_generate(llm, requests, params):
    # Batched generate; if the batch fails, retry one by one so one bad prompt cannot sink it.
    try:
        outs = llm.generate(requests, params, use_tqdm=False)
    except Exception as exc:
        if len(requests) == 1:
            return [Generation(text="", error=repr(exc)[:800])]
        return [g for r in requests for g in vllm_generate(llm, [r], params)]
    res = []
    for o in outs:
        c = o.outputs[0]
        res.append(Generation(text=c.text, n_prompt_tokens=len(o.prompt_token_ids or []),
                              n_output_tokens=len(c.token_ids), finish_reason=c.finish_reason))
    return res


def hf_generate_one(model, processor, messages, max_tokens, torch):
    # Single-request transformers generate (no padding issues; slow fallback).
    try:
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        n_in = int(inputs["input_ids"].shape[1])
        new = ids[0, n_in:]
        text = processor.decode(new, skip_special_tokens=True)
        return Generation(text=text, n_prompt_tokens=n_in, n_output_tokens=int(new.shape[0]),
                          finish_reason="length" if new.shape[0] >= max_tokens else "stop")
    except Exception as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return Generation(text="", error=repr(exc)[:800])


def content_list(prompt, with_images):
    # OpenAI/HF-style interleaved content in Prompt.parts order.
    out = []
    for kind, val in prompt.parts:
        if kind == "text":
            out.append({"type": "text", "text": val})
        else:
            out.append({"type": "image", "image": val} if with_images else {"type": "image"})
    return out


def dump_info(obj):
    return json.loads(json.dumps(obj, default=str))
