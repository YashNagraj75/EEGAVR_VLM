from .base import VLM, sampling_info
from ._common import content_list, hf_generate_one, model_revision, versions, vllm_generate

REPO_VLLM = "OpenGVLab/InternVL3_5-8B"
REPO_HF = "OpenGVLab/InternVL3_5-8B-HF"
STOP_TOKENS = ("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>")


def to_text(prompt):
    # vLLM InternVL expects one "<image>" placeholder per image, in order.
    return "".join(val if kind == "text" else "<image>" for kind, val in prompt.parts)


class VllmModel(VLM):
    # vLLM engine for InternVL3.5 (original-format repo); unverified without a GPU.
    tag, spec_key = "internvl3.5-8b", "internvl"

    def __init__(self, path, max_tokens=256, max_model_len=16384, gpu_memory_utilization=0.90,
                 max_num_seqs=16, seed=0, **_):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        self.path, self.max_tokens = path, max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.llm = LLM(model=path, trust_remote_code=True, dtype="bfloat16", max_model_len=max_model_len,
                       seed=seed, gpu_memory_utilization=gpu_memory_utilization, max_num_seqs=max_num_seqs,
                       limit_mm_per_prompt={"image": 14}, mm_processor_kwargs={"max_dynamic_patch": 12})
        stop_ids = []
        for t in STOP_TOKENS:
            i = self.tokenizer.convert_tokens_to_ids(t)
            if i is not None and i != self.tokenizer.unk_token_id:
                stop_ids.append(i)
        self.params = SamplingParams(temperature=0.0, max_tokens=max_tokens, stop_token_ids=stop_ids)
        self.engine_args = dict(max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
                                max_num_seqs=max_num_seqs, seed=seed, stop_token_ids=stop_ids)

    def request(self, prompt):
        msgs = [{"role": "user", "content": to_text(prompt)}]
        text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        req = {"prompt": text}
        if prompt.images:
            req["multi_modal_data"] = {"image": prompt.images}
        return req

    def generate(self, prompts):
        return vllm_generate(self.llm, [self.request(p) for p in prompts], self.params)

    def info(self):
        return {"tag": self.tag, "backend": "vllm", "repo": REPO_VLLM, "path": self.path,
                "revision": model_revision(self.path, REPO_VLLM), "versions": versions(),
                "sampling": sampling_info(self.max_tokens), "engine": self.engine_args}


class HFModel(VLM):
    # transformers fallback on the -HF checkpoint (same weights, native classes); batch size 1.
    tag, spec_key = "internvl3.5-8b", "internvl"

    def __init__(self, path, max_tokens=256, **_):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch, self.path, self.max_tokens = torch, path, max_tokens
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()

    def generate(self, prompts):
        msgs = lambda p: [{"role": "user", "content": content_list(p, True)}]
        return [hf_generate_one(self.model, self.processor, msgs(p), self.max_tokens, self.torch) for p in prompts]

    def info(self):
        return {"tag": self.tag, "backend": "hf", "repo": REPO_HF, "path": self.path,
                "revision": model_revision(self.path, REPO_HF), "versions": versions(),
                "sampling": sampling_info(self.max_tokens)}
