MODEL_IDS = {
    "qwen3vl": {"vllm": "Qwen/Qwen3-VL-8B-Instruct", "hf": "Qwen/Qwen3-VL-8B-Instruct"},
    "internvl": {"vllm": "OpenGVLab/InternVL3_5-8B", "hf": "OpenGVLab/InternVL3_5-8B-HF"},
}
TAGS = {"qwen3vl": "qwen3vl-8b", "internvl": "internvl3.5-8b"}


def get_model(name, backend="vllm", model_path=None, **kw):
    # Imports are lazy so the harness, tests and dry-runs work without torch/vLLM installed.
    if name == "mock":
        from .mock import MockVLM
        return MockVLM(**kw)
    path = model_path or MODEL_IDS[name][backend]
    if name == "qwen3vl":
        from . import qwen3vl as mod
    elif name == "internvl":
        from . import internvl as mod
    else:
        raise ValueError(f"unknown model {name!r}")
    cls = mod.VllmModel if backend == "vllm" else mod.HFModel
    return cls(path, **kw)
