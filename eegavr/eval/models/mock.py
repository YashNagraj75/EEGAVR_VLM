import hashlib
import json
import random

from .base import VLM, Generation

MALFORMED = [
    '```json\n{"T1": "left", "T2": "right", "rule": "x"}\n```',
    "{'T1': 'Right', 'T2': 'left', 'rule': 'single quotes',}",
    "I think T1 belongs to the left group and T2 is right.",
    '{"T1": "left|right", "T2": "left|right", "rule": "<one sentence>"}',
    '{"T1": "left", "rule": "only one panel"}',
    "",
    '<think>T1 left? no</think>{"T1": "RIGHT", "T2": "LEFT", "rule": "thinking stripped"}',
    '{"T1": "left", "T2": "right", "rule": "truncated',
    "Both panels go to the left.",
]


class MockVLM(VLM):
    # Modes: left | random | oracle | malformed | crash_after:N. Validates prompt structure.
    max_images = 14

    def __init__(self, mode="random", answers=None, seed=0):
        self.mode, self.answers, self.seed, self.calls = mode, answers or {}, seed, 0
        self.tag = f"mock-{mode.split(':')[0]}"
        self.spec_key = "mock"
        self.seen = []

    def _check(self, prompt):
        text = prompt.text_with_placeholders()
        n_img = len(prompt.images)
        assert n_img in (0, 1, 14), f"unexpected image count {n_img}"
        for lab in ["L1", "L6", "R1", "R6", "T1", "T2"]:
            assert lab in text, f"label {lab} missing from prompt"
        assert prompt.meta["problem_id"] not in text, "problem id leaked into prompt"
        return text

    def _answer(self, prompt, text):
        if self.mode == "left":
            return '{"T1": "left", "T2": "left", "rule": "always left"}'
        if self.mode == "oracle":
            a = self.answers[prompt.meta["problem_id"]]
            return json.dumps({"T1": a[0], "T2": a[1], "rule": "oracle"})
        if self.mode == "malformed":
            return MALFORMED[(self.calls - 1) % len(MALFORMED)]
        h = int(hashlib.md5((str(self.seed) + text).encode()).hexdigest(), 16)
        rng = random.Random(h)
        return json.dumps({"T1": rng.choice(["left", "right"]), "T2": rng.choice(["left", "right"]), "rule": "random"})

    def generate(self, prompts):
        outs = []
        for p in prompts:
            self.calls += 1
            if self.mode.startswith("crash_after:") and self.calls > int(self.mode.split(":")[1]):
                raise RuntimeError("mock crash")
            text = self._check(p)
            self.seen.append(p)
            ans = self._answer(p, text)
            outs.append(Generation(text=ans, n_prompt_tokens=len(text) // 4 + 260 * len(p.images),
                                   n_output_tokens=len(ans) // 4, finish_reason="stop"))
        return outs

    def info(self):
        return {"tag": self.tag, "backend": "mock", "mode": self.mode}
