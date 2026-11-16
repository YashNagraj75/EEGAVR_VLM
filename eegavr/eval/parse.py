import json
import re

# Parse statuses from best to worst; partial/unparsable/error panels score as wrong.
STATUSES = ("json", "json_repaired", "regex", "partial", "unparsable", "error")

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_KEY_ALIASES = {"t1": "T1", "test1": "T1", "testpanel1": "T1", "t2": "T2", "test2": "T2", "testpanel2": "T2"}


def _side(value):
    if not isinstance(value, str):
        return None
    v = value.strip().lower().strip(" .\"'`")
    has_l, has_r = bool(re.search(r"\bleft\b", v)), bool(re.search(r"\bright\b", v))
    if has_l != has_r:
        return "left" if has_l else "right"
    if v in ("l", "left group"):
        return "left"
    if v in ("r", "right group"):
        return "right"
    return None


def _objects(text):
    # Every balanced {...} substring, outermost first, in order of appearance.
    out, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                out.append(text[start:i + 1])
    return out


def _repair(s):
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = re.sub(r"(?<=[{,\s])'([^'\n]*?)'\s*:", r'"\1":', s)
    s = re.sub(r":\s*'([^'\n]*?)'", r': "\1"', s)
    s = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_ ]*?)\s*:", r'\1"\2":', s)
    return s


def _from_dict(d):
    res = {}
    for k, v in d.items():
        key = _KEY_ALIASES.get(re.sub(r"[\s_\-]", "", str(k)).lower())
        if key:
            res[key] = _side(v)
        elif str(k).strip().lower() == "rule" and isinstance(v, str):
            res["rule"] = v.strip()
    return res


def parse_answer(raw):
    # -> dict(T1, T2 in {"left","right",None}, rule, status)
    text = _THINK.sub("", raw or "")
    text = re.sub(r"```(?:json)?", "", text, flags=re.I)
    best = None
    for repaired in (False, True):
        for obj in reversed(_objects(text)):
            try:
                d = json.loads(_repair(obj) if repaired else obj)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(d, dict):
                continue
            got = _from_dict(d)
            if got.get("T1") and got.get("T2"):
                return {"T1": got["T1"], "T2": got["T2"], "rule": got.get("rule"),
                        "status": "json_repaired" if repaired else "json"}
            if best is None and (got.get("T1") or got.get("T2")):
                best = got
    found = {}
    for lab in ("T1", "T2"):
        filler = r"(?:[^\w\n]|\b(?:is|belongs|goes|to|on|in|the|group|side|panel|should|be)\b)"
        hits = re.findall(rf"\b{lab[0]}\s*-?\s*{lab[1]}\b{filler}{{0,12}}(left|right)\b(?!\s*[|/])", text, flags=re.I)
        sides = {h.lower() for h in hits}
        if len(sides) == 1:
            found[lab] = sides.pop()
    if best:
        for lab in ("T1", "T2"):
            found.setdefault(lab, best.get(lab))
    if found.get("T1") and found.get("T2"):
        return {"T1": found["T1"], "T2": found["T2"], "rule": (best or {}).get("rule"), "status": "regex"}
    if found.get("T1") or found.get("T2"):
        return {"T1": found.get("T1"), "T2": found.get("T2"), "rule": (best or {}).get("rule"), "status": "partial"}
    return {"T1": None, "T2": None, "rule": None, "status": "unparsable"}
