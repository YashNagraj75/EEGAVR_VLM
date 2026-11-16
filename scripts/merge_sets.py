import argparse
import json
import os
import shutil
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eegavr.generator.predicates import truth
from eegavr.generator.version import GENERATOR_VERSION


def main():
    # new set = base set + added rules: panels/masks hard-linked, scenes rewritten with the full truth dict
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("add")
    ap.add_argument("out")
    a = ap.parse_args()
    for d in ("panels", "masks"):
        os.makedirs(f"{a.out}/{d}", exist_ok=True)
        for src in (a.base, a.add):
            for f in os.listdir(f"{src}/{d}"):
                os.link(f"{src}/{d}/{f}", f"{a.out}/{d}/{f}")
    os.makedirs(f"{a.out}/scenes", exist_ok=True)
    for src in (a.base, a.add):
        for f in os.listdir(f"{src}/scenes"):
            sc = json.load(open(f"{src}/scenes/{f}"))
            sc["truth"] = {k: bool(v) for k, v in truth(sc).items()}
            sc.pop("_polys", None)
            json.dump(sc, open(f"{a.out}/scenes/{f}", "w"))
    for d in ("problems", "audit"):
        os.makedirs(f"{a.out}/{d}", exist_ok=True)
        for src in (a.base, a.add):
            if os.path.isdir(f"{src}/{d}"):
                for f in os.listdir(f"{src}/{d}"):
                    shutil.copy(f"{src}/{d}/{f}", f"{a.out}/{d}/{f}")
    os.makedirs(f"{a.out}/splits", exist_ok=True)
    for f in sorted(set(os.listdir(f"{a.base}/splits")) | set(os.listdir(f"{a.add}/splits"))):
        ids = []
        for src in (a.base, a.add):
            if os.path.exists(f"{src}/splits/{f}"):
                ids += open(f"{src}/splits/{f}").read().split()
        open(f"{a.out}/splits/{f}", "w").write("\n".join(sorted(ids)) + ("\n" if ids else ""))
    mb, ma = json.load(open(f"{a.base}/manifest.json")), json.load(open(f"{a.add}/manifest.json"))
    m = dict(mb)
    m["generator_version"] = GENERATOR_VERSION
    m["git_commit"] = ma["git_commit"]
    m["base_set"] = {"generator_version": mb["generator_version"], "git_commit": mb["git_commit"]}
    m["rules"] = {**mb["rules"], **ma["rules"]}
    for k in ("assembly_stats", "audit_status"):
        m[k] = {**mb.get(k, {}), **ma.get(k, {})}
    m["counts"] = {sp: {**mb["counts"].get(sp, {}), **ma["counts"].get(sp, {})} for sp in mb["counts"]}
    m["skipped_rule_splits"] = sorted(set(mb.get("skipped_rule_splits", [])) | set(ma.get("skipped_rule_splits", [])))
    m["dropped"] = {**mb.get("dropped", {}), **ma.get("dropped", {})}
    m["n_panels"] = mb["n_panels"] + ma["n_panels"]
    m["n_problems"] = mb["n_problems"] + ma["n_problems"]
    m["truth_keys"] = sorted(truth({"objects": []}).keys())
    json.dump(m, open(f"{a.out}/manifest.json", "w"), indent=1)
    print("merged", m["n_panels"], "panels", m["n_problems"], "problems")


if __name__ == "__main__":
    main()
