import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

SUPPORTED_CONTRACT_VERSIONS = {1, "1", "v1"}
SPLITS = ("validation", "test_id", "ood_count", "ood_count_control", "ood_render")

# Problem-level fields that are scoring/analysis only (contract "What the harness may read",
# plus answer-bearing fields the contract implies: positive_side, test[*].answer).
SCORING_ONLY_KEYS = frozenset({
    "rule_id", "type", "rule_setting_id", "knob", "partner_problem_id", "twin_map",
    "positive_side", "split", "problem_id",
})
# Keys that must never occur anywhere inside described/symbolic payloads.
FORBIDDEN_NESTED_KEYS = frozenset({
    "role", "truth", "rule_id", "rule_setting_id", "twin_map", "knob", "positive_side",
    "answer", "partner_problem_id", "panel_id", "seed",
})


class LeakageError(ValueError):
    pass


class ContractError(ValueError):
    pass


@dataclass(frozen=True)
class PromptView:
    # The only problem content a prompt builder may see. Panel ids are kept for image lookup
    # and must never be rendered into prompt text (they can encode the rule id).
    left: tuple
    right: tuple
    test: tuple
    described: dict
    symbolic: dict


def _walk_keys(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, f"{path}/{k}"
            yield from _walk_keys(v, f"{path}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk_keys(v, f"{path}[{i}]")


def _check_payload(problem, field, panel_ids):
    payload = problem.get(field) or {}
    for pid, item in payload.items():
        for key, where in _walk_keys(item):
            if key in FORBIDDEN_NESTED_KEYS:
                raise LeakageError(f"{problem.get('problem_id')}: forbidden key '{key}' at {field}/{pid}{where}")
        text = item if isinstance(item, str) else json.dumps(item)
        ids = set(panel_ids) | {problem.get("problem_id", "")}
        for other in ids:
            if other and other in text:
                raise LeakageError(f"{problem.get('problem_id')}: identifier '{other}' appears in {field}/{pid}")


def validate_problem(problem):
    # Structural checks against data contract v1; raises ContractError on violation.
    pid = problem.get("problem_id", "?")
    for key in ("problem_id", "left", "right", "test"):
        if key not in problem:
            raise ContractError(f"{pid}: missing '{key}'")
    if len(problem["left"]) != 6 or len(problem["right"]) != 6:
        raise ContractError(f"{pid}: expected 6+6 support panels, got {len(problem['left'])}+{len(problem['right'])}")
    if len(problem["test"]) != 2:
        raise ContractError(f"{pid}: expected 2 test panels, got {len(problem['test'])}")
    for t in problem["test"]:
        if t.get("answer") not in ("left", "right"):
            raise ContractError(f"{pid}: test answer must be left/right, got {t.get('answer')!r}")
    ids = list(problem["left"]) + list(problem["right"]) + [t["panel_id"] for t in problem["test"]]
    if len(set(ids)) != 14:
        raise ContractError(f"{pid}: the 14 panel ids are not unique")
    return ids


def prompt_view(problem, need_described=False, need_symbolic=False):
    # Strips every scoring-only field; checks text payloads for leaks.
    ids = validate_problem(problem)
    described = dict(problem.get("described") or {})
    symbolic = dict(problem.get("symbolic") or {})
    for need, field, d in ((need_described, "described", described), (need_symbolic, "symbolic", symbolic)):
        if need:
            missing = [i for i in ids if i not in d]
            if missing:
                raise ContractError(f"{problem['problem_id']}: {field} missing for {len(missing)} panels")
    _check_payload(problem, "described", ids)
    _check_payload(problem, "symbolic", ids)
    return PromptView(
        left=tuple(problem["left"]), right=tuple(problem["right"]),
        test=tuple(t["panel_id"] for t in problem["test"]),
        described={i: described[i] for i in ids if i in described},
        symbolic={i: symbolic[i] for i in ids if i in symbolic},
    )


class EvalSet:
    def __init__(self, root):
        self.root = Path(root)
        mpath = self.root / "manifest.json"
        self.manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
        cv = self.manifest.get("contract_version")
        if cv is not None and cv not in SUPPORTED_CONTRACT_VERSIONS:
            raise ContractError(f"unsupported contract_version {cv!r}")
        self._problem_cache = {}

    def splits(self):
        d = self.root / "splits"
        return sorted(p.stem for p in d.glob("*.txt")) if d.exists() else []

    def split_ids(self, split):
        path = self.root / "splits" / f"{split}.txt"
        if not path.exists():
            raise FileNotFoundError(f"split file not found: {path}")
        return [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]

    def load_problem(self, problem_id):
        if problem_id not in self._problem_cache:
            path = self.root / "problems" / f"{problem_id}.json"
            self._problem_cache[problem_id] = json.loads(path.read_text())
            if len(self._problem_cache) > 4096:
                self._problem_cache.pop(next(iter(self._problem_cache)))
        return self._problem_cache[problem_id]

    def panel_path(self, panel_id):
        return self.root / "panels" / f"{panel_id}.png"

    def load_panel(self, panel_id):
        with Image.open(self.panel_path(panel_id)) as im:
            im.load()
            return im.convert("L")

    def digest(self):
        # md5 over manifest + split files: identifies the frozen set in every log record.
        h = hashlib.md5()
        for p in [self.root / "manifest.json", *sorted((self.root / "splits").glob("*.txt"))]:
            if p.exists():
                h.update(p.name.encode())
                h.update(p.read_bytes())
        return h.hexdigest()


def select_problems(evalset, split, max_problems=None):
    # Split-file order; with a cap, round-robin over rules so every rule keeps equal weight.
    ids = evalset.split_ids(split)
    if not max_problems or max_problems >= len(ids):
        return ids
    by_rule = {}
    for pid in ids:
        by_rule.setdefault(evalset.load_problem(pid).get("rule_id", "?"), []).append(pid)
    out, queues = [], [list(v) for _, v in sorted(by_rule.items())]
    while len(out) < max_problems and any(queues):
        for q in queues:
            if q and len(out) < max_problems:
                out.append(q.pop(0))
    order = {pid: i for i, pid in enumerate(ids)}
    return sorted(out, key=order.__getitem__)
