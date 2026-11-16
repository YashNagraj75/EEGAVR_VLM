from dataclasses import dataclass


@dataclass
class Generation:
    text: str
    n_prompt_tokens: int = None
    n_output_tokens: int = None
    finish_reason: str = None
    error: str = None


class VLM:
    # Adapters turn Prompt.parts into model requests; they must not read Prompt.meta.
    tag = "base"
    spec_key = "mock"
    max_images = 14

    def generate(self, prompts):
        raise NotImplementedError

    def info(self):
        return {"tag": self.tag}

    def close(self):
        pass


def sampling_info(max_tokens):
    return {"decoding": "greedy", "temperature": 0.0, "max_tokens": max_tokens}
