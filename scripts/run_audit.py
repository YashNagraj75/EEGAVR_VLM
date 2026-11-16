import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eegavr.audit.run import audit_setting


def main():
    # audit the given rule settings (5a, 5b, 5d and the 5c reference pool)
    ap = argparse.ArgumentParser()
    ap.add_argument("settings", nargs="+")
    ap.add_argument("--pairs", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tuning", default="{}")
    ap.add_argument("--out", default="runs/audit")
    a = ap.parse_args()
    for sid in a.settings:
        r = audit_setting(sid, a.pairs, a.workers, json.loads(a.tuning), a.out)
        pa, pf, b = r["5a"]["pass_set"], r["5a"]["full_set"], r["5b"]
        print(f"{sid:10s} kept={r['kept_rate']:.3f} 5a={pa['best']}:{pa['best_acc']:.3f} "
              f"[{pa['best_ci90'][0]:.3f},{pa['best_ci90'][1]:.3f}] full={pf['best_acc']:.3f} "
              f"5b={b['best_acc']:.3f}[{b['best_ci90'][0]:.3f},{b['best_ci90'][1]:.3f}] "
              f"5d224={r['5d']['224']['all_ok'] if '224' in r['5d'] else r['5d'][224]['all_ok']:.3f} "
              f"pass={r['pass']} {r['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
