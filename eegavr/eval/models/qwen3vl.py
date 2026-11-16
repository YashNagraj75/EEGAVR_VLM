from .base import VLM, sampling_info
from ._common import content_list, hf_generate_one, model_revision, versions, vllm_generate

REPO = "Qwen/Qwen3-VL-8B-Instruct"


def to_messages(prompt, with_images=False):
    return [{"role": "user", "content": content_list(prompt, with_images)}]


class VllmModel(VLM):
    # GPU-host only (unverified in the sandbox): vLLM offline engine, greedy, multi-image.
    tag, spec_key = "qwen3vl-8b", "qwen3vl"

    def __init__(self, path, max_tokens=256, max_model_len=16384, gpu_memory_utilization=0.90,
                 max_num_seqs=16, seed=0, **_):
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        self.path, self.max_tokens = path, max_tokens
        self.processor = AutoProcessor.from_pretrained(path)
        self.llm = LLM(model=path, dtype="bfloat16", max_model_len=max_model_len, seed=seed,
                       gpu_memory_utilization=gpu_memory_utilization, max_num_seqs=max_num_seqs,
                       limit_mm_per_prompt={"image": 14, "video": 0})
        self.params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        self.engine_args = dict(max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
                                max_num_seqs=max_num_seqs, seed=seed)

    def request(self, prompt):
        text = self.processor.apply_chat_template(to_messages(prompt), tokenize=False, add_generation_prompt=True)
        req = {"prompt": text}
        if prompt.images:
            req["multi_modal_data"] = {"image": prompt.images}
        return req

    def generate(self, prompts):
        return vllm_generate(self.llm, [self.request(p) for p in prompts], self.params)

    def info(self):
        return {"tag": self.tag, "backend": "vllm", "repo": REPO, "path": self.path,
                "revision": model_revision(self.path, REPO), "versions": versions(),
                "sampling": sampling_info(self.max_tokens), "engine": self.engine_args}


class HFModel(VLM):
    # transformers fallback (>=4.57 has Qwen3VLForConditionalGeneration); batch size 1, slow.
    tag, spec_key = "qwen3vl-8b", "qwen3vl"

    def __init__(self, path, max_tokens=256, **_):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.torch, self.path, self.max_tokens = torch, path, max_tokens
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            path, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval()

    def generate(self, prompts):
        return [hf_generate_one(self.model, self.processor, to_messages(p, True), self.max_tokens, self.torch)
                for p in prompts]

    def info(self):
        return {"tag": self.tag, "backend": "hf", "repo": REPO, "path": self.path,
                "revision": model_revision(self.path, REPO), "versions": versions(),
                "sampling": sampling_info(self.max_tokens)}
