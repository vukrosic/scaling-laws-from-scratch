"""Every number the film shows, written to values.json.

    python values.py

Reads files/runs.jsonl (made by sweep.py). Trains one model of
its own, once, to record the 27 chances it gives inside one
sentence of the test letters; that is kept in files/chances.json.
"""
import json
import math
import os

import torch

from laws import (best_runs, best_split, column, fit_floor, fit_law,
                  fit_power, formula, load, same_work, table)
from model import Model
from step5_text import floors
from step6_formula import misses, points
from step7_budget import BUDGETS
from step8_schedule import (ALL, on_the_way, planned, size_share,
                            way_points)
from step9_checks import dice, shapes
from sweep import STEPS, WIDTHS, write
from train import STEP, TEST_FROM, run, test_loss, text

NAMES = " abcdefghijklmnopqrstuvwxyz"
LETTERS = 977 * STEP
FIT = 5
SENTENCE = ("maine is the only state that borders exactly one other "
            "state maine is the easternmost state in the contiguous "
            "four eight states")
CHANCES = "files/chances.json"
device = "mps" if torch.backends.mps.is_available() else "cpu"


def find_sentence():
    """Where the sentence starts, in the test letters."""
    book = text()
    want = torch.tensor([NAMES.index(c) for c in SENTENCE])
    part = book[TEST_FROM:TEST_FROM + 1_000_000]
    for i in (part == want[0]).nonzero().flatten().tolist():
        if torch.equal(part[i:i + len(want)], want.to(part.dtype)):
            return TEST_FROM + i
    raise SystemExit("sentence not found")


def chances():
    """The 27 chances a trained model gives before every letter of
    the sentence. Trained once (width 128, 4 million letters)."""
    if os.path.exists(CHANCES):
        return json.load(open(CHANCES))
    won = [r for r in best_runs(load())
           if (r["width"], r["layers"], r["steps"], r["seed"])
           == (128, 2, 977, 0) and "budget" not in r]
    result = run(128, 2, LETTERS, 0, device, won[0]["peak"], keep=True)
    model = result["model"]
    start = find_sentence()
    book = text()
    rows = []
    with torch.no_grad():
        for i in range(len(SENTENCE)):
            seen = book[start + i - 64:start + i].long()[None]
            out = model(seen.to(device))[0, -1].cpu()
            rows.append([round(float(c), 5)
                         for c in torch.softmax(out, dim=0)])
    found = {"start": start, "rows": rows,
             "score": result["loss"] / math.log(2)}
    json.dump(found, open(CHANCES, "w"))
    return found


def main():
    runs = load()
    book = text()
    out = {}

    # ---- the text and the two guessers that need no training
    count = torch.bincount(book[:TEST_FROM], minlength=27).double()
    share = count / count.sum()
    test = book[TEST_FROM:].long()
    out["text"] = {
        "total": len(book), "test": len(book) - TEST_FROM,
        "step": STEP, "start": "".join(NAMES[i] for i in book[:400]),
        "sentence": SENTENCE,
    }
    out["letters"] = {
        "share": [round(float(s), 5) for s in share],
        "blind": math.log2(27),
        "count": -torch.log2(share[test]).mean().item(),
    }
    out["chances"] = chances()

    # ---- the smallest model, before and after
    torch.manual_seed(0)
    fresh = Model(4, 2).to(device)
    out["fresh"] = {
        "size": fresh.size(),
        "score": test_loss(fresh, device) / math.log(2),
        "sample": write(fresh, device, 160),
    }

    # ---- eight sizes at 4 million letters
    best = {(r["width"], r["steps"]): r for r in best_runs(runs)
            if r["layers"] == 2 and r["seed"] == 0
            and "budget" not in r}
    out["sizes"] = [{
        "width": w, "size": best[(w, 977)]["size"],
        "score": best[(w, 977)]["loss"],
        "sample": best[(w, 977)]["sample"],
    } for w in WIDTHS]
    rows = column(runs, LETTERS)
    sizes = [s for s, _ in rows]
    scores = [v for _, v in rows]
    floor, power, scale = fit_floor(sizes[:FIT], scores[:FIT])
    power0, scale0 = fit_power(sizes[:FIT], scores[:FIT])
    out["rule"] = {
        "fit": FIT, "floor": floor, "power": power, "scale": scale,
        "share": [scores[i] / scores[i - 1] for i in range(1, FIT)],
        "gaps": [s - floor for s in scores],
        "x4": 4 ** power, "x10": 10 ** power,
        "half": 0.5 ** (1 / power),
        "forecast": [floor + scale * s ** power for s in sizes],
        "plain": [scale0 * s ** power0 for s in sizes],
        "reach": sizes[-1] / sizes[FIT - 1],
    }

    # ---- every size on every amount of text (step 5)
    grid = table(runs)
    out["grid"] = {
        "letters": [s * STEP for s in STEPS],
        "scores": [[grid.get((size, s * STEP)) for s in STEPS]
                   for size in sizes],
    }
    out["floors"] = [
        {"letters": letters, "floor": level, "power": p, "scale": k}
        for letters, level, p, k in floors(runs)]

    # ---- one formula for size and text (step 6)
    measured = points(runs)
    law = fit_law(measured)
    off = misses(law, measured)
    _, a, alpha, b, beta = law
    out["law"] = {
        "floor": law[0], "a": a, "alpha": alpha, "b": b, "beta": beta,
        "runs": len(measured),
        "x4size": 4 ** -alpha, "x4text": 4 ** -beta,
        "miss": (sum(m * m for m in off) / len(off)) ** 0.5,
        "worst": max(abs(m) for m in off),
        "share": beta / (alpha + beta),
    }

    # ---- equal work, different splits (step 7)
    out["budget"] = []
    for work in BUDGETS:
        size, letters, loss = best_split(work, law)
        out["budget"].append({
            "work": work,
            "runs": [{"size": r["size"], "letters": r["letters"],
                      "score": r["loss"],
                      "formula": formula(r["size"], r["letters"], law)}
                     for r in same_work(runs, work)],
            "best": {"size": size, "letters": letters, "score": loss},
        })

    # ---- read off on the way, or planned for the length (step 8)
    out["plan"] = [{
        "size": planned(runs, w, 977)[0],
        "marks": [{"letters": own * STEP,
                   "way": on_the_way(runs, w, mark),
                   "own": (planned(runs, w, own) or [0, None])[1]}
                  for mark, own in ALL],
    } for w in WIDTHS]

    way = fit_law(way_points(runs))
    out["way"] = {"share": size_share(way), "own": size_share(law),
                  "runs": len(way_points(runs))}

    # ---- shape and dice (step 9)
    out["shapes"] = shapes(runs)
    out["dice"] = dice(runs)

    json.dump(out, open("values.json", "w"), indent=1)
    print("values.json written:", ", ".join(out))


if __name__ == "__main__":
    main()
