import argparse
import json
import os
import numpy as np
from PIL import Image, ImageDraw


def sheet(root, pid, out):
    # one problem: left 2x3, right 2x3, then the two test panels (no labels revealing the answer)
    p = json.load(open(f"{root}/problems/{pid}.json"))
    S, g = 200, 12
    img = lambda i: Image.open(f"{root}/panels/{i}.png").convert("L").resize((S, S), Image.BOX)
    W = 2 * (3 * S + 2 * g) + 3 * g + 2 * S + 4 * g
    H = 2 * S + g + 40
    c = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(c)
    x0 = 0
    for side in ("left", "right"):
        d.text((x0 + 4, 4), side.upper(), fill=0)
        for k, i in enumerate(p[side]):
            x, y = x0 + (k % 3) * (S + g), 30 + (k // 3) * (S + g)
            c.paste(img(i), (x, y))
            d.rectangle([x, y, x + S - 1, y + S - 1], outline=150)
        x0 += 3 * S + 2 * g + 3 * g
    x0 += 2 * g
    d.text((x0 + 4, 4), "TEST", fill=0)
    for k, t in enumerate(p["test"]):
        y = 30 + k * (S + g)
        c.paste(img(t["panel_id"]), (x0, y))
        d.rectangle([x0, y, x0 + S - 1, y + S - 1], outline=150)
    c.save(out)
    return p


def main():
    # first n test_id problems per rule, stacked, written as <out>/contact_<rule>.png
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ids = open(f"{a.root}/splits/test_id.txt").read().split()
    seen = {}
    for i in ids:
        seen.setdefault(i.rsplit("-", 2)[0], []).append(i)
    for key, group in sorted(seen.items()):
        rows = []
        for k, i in enumerate(group[:a.n]):
            p = sheet(a.root, i, f"{a.out}/_row{k}.png")
            rows.append(Image.open(f"{a.out}/_row{k}.png"))
            os.remove(f"{a.out}/_row{k}.png")
            print(key, i, p["positive_side"], [t["answer"] for t in p["test"]])
        c = Image.new("L", (rows[0].width, sum(r.height for r in rows)), 255)
        for k, r in enumerate(rows):
            c.paste(r, (0, k * rows[0].height))
        c.save(f"{a.out}/contact_{key}.png")

if __name__ == "__main__":
    main()
