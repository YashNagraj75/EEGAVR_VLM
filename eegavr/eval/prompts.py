import json
from dataclasses import dataclass, field

from .dataset import prompt_view
from .vision import prepare_separate, render_composite

PROMPT_VERSION = "neutral-v1"
CONDITIONS = ("visual", "described", "symbolic")
LAYOUTS = ("separate", "composite")
LEFT_LABELS = tuple(f"L{i}" for i in range(1, 7))
RIGHT_LABELS = tuple(f"R{i}" for i in range(1, 7))
TEST_LABELS = ("T1", "T2")

INTRO = (
    "This is a visual reasoning puzzle. There are two groups of example panels: a LEFT group "
    "(L1-L6) and a RIGHT group (R1-R6). All panels in one group follow a common rule, and none "
    "of the panels in the other group follow it; the rule may hold on the left and fail on the "
    "right, or the other way round. There are also two test panels, T1 and T2. Work out the rule, "
    "then decide for each test panel whether it belongs to the LEFT group or the RIGHT group."
)
PRESENT = {
    ("visual", "separate"): "The panels are shown as separate images, each one preceded by its label.",
    ("visual", "composite"): (
        "All panels are shown in one image. The LEFT group (L1-L6) is the block at the top left, the "
        "RIGHT group (R1-R6) is the block at the top right, and the test panels T1 and T2 are below "
        "them. Every panel is labelled."
    ),
    ("described", None): (
        "Each panel is described in words. Positions are fractions of the panel width: x is measured "
        "from the left edge and y from the top edge, so a smaller y means higher up. Size is the "
        "radius of a shape as a fraction of the panel width; rotations are in degrees, counter-clockwise."
    ),
    ("symbolic", None): (
        "Each panel is given as a JSON scene description with a list of objects and a list of "
        "relations. For each object: shape, fill, centre x and y as fractions of the panel width "
        "(x from the left edge, y from the top edge, so a smaller y means higher up), size (radius as "
        "a fraction of the panel width) and rotation (degrees, counter-clockwise). A relation "
        "[name, i, j] means object i has relation name to object j (for example [\"above\", 1, 2] "
        "means object 1 is above object 2); [\"between\", m, a, b] means object m lies between "
        "objects a and b. Only relations that hold are listed."
    ),
}
ANSWER = (
    "Answer with only a JSON object and no other text, in exactly this format:\n"
    '{"T1": "left|right", "T2": "left|right", "rule": "<one sentence>"}\n'
    'where each of T1 and T2 is either "left" or "right", and rule states the rule in one sentence.'
)


@dataclass
class Prompt:
    # Model-agnostic prompt: ordered ("text", str) / ("image", PIL.Image) parts.
    # meta is for logging only; model adapters must read `parts` and nothing else.
    parts: list
    meta: dict = field(default_factory=dict)

    @property
    def images(self):
        return [p for kind, p in self.parts if kind == "image"]

    def text_with_placeholders(self, placeholder="<image>"):
        return "".join(p if kind == "text" else placeholder for kind, p in self.parts)


def label_map(view):
    m = dict(zip(LEFT_LABELS, view.left))
    m.update(zip(RIGHT_LABELS, view.right))
    m.update(zip(TEST_LABELS, view.test))
    return m


def _panel_text(condition, view, pid):
    if condition == "described":
        return str(view.described[pid]).strip()
    return json.dumps(view.symbolic[pid], separators=(", ", ": "), ensure_ascii=True)


def build_prompt(problem, condition, layout, spec, load_panel):
    # load_panel(panel_id) -> PIL image; called only for visual prompts, in display order.
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    if condition == "visual" and layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}")
    view = prompt_view(problem, need_described=condition == "described", need_symbolic=condition == "symbolic")
    labels = label_map(view)
    parts = [("text", INTRO + "\n\n")]
    groups = (("LEFT group:", LEFT_LABELS), ("RIGHT group:", RIGHT_LABELS), ("Test panels:", TEST_LABELS))
    if condition == "visual" and layout == "separate":
        parts.append(("text", PRESENT[("visual", "separate")] + "\n"))
        for header, labs in groups:
            parts.append(("text", f"\n{header}\n"))
            for lab in labs:
                parts.append(("text", f"{lab}: "))
                parts.append(("image", prepare_separate(load_panel(labels[lab]), spec)))
                parts.append(("text", "\n"))
    elif condition == "visual":
        parts.append(("text", PRESENT[("visual", "composite")] + "\n"))
        parts.append(("image", render_composite({lab: load_panel(pid) for lab, pid in labels.items()}, spec)))
        parts.append(("text", "\n"))
    else:
        parts.append(("text", PRESENT[(condition, None)] + "\n"))
        for header, labs in groups:
            parts.append(("text", f"\n{header}\n"))
            for lab in labs:
                parts.append(("text", f"{lab}: {_panel_text(condition, view, labels[lab])}\n"))
    parts.append(("text", "\n" + ANSWER))
    merged = []
    for kind, val in parts:
        if kind == "text" and merged and merged[-1][0] == "text":
            merged[-1] = ("text", merged[-1][1] + val)
        else:
            merged.append((kind, val))
    meta = {"problem_id": problem["problem_id"], "condition": condition,
            "layout": layout if condition == "visual" else "none", "prompt_version": PROMPT_VERSION,
            "image_sizes": [list(im.size) for kind, im in merged if kind == "image"]}
    return Prompt(parts=merged, meta=meta)
