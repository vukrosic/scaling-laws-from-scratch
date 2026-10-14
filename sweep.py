"""Train every model size on every amount of text. Takes hours.

    python sweep.py mps 4 8 16 32 64     # these widths, on the GPU
    python sweep.py mps 128 256 384
    python sweep.py mps 46 90            # two more shapes
    python sweep.py mps budget           # runs of equal work
    python sweep.py mps dice             # same runs, other dice

Every finished run is one line in files/runs.jsonl, so the sweep
can be stopped and started again. laws.py reads that file.
"""
import json
import math
import os
import sys

import torch

from model import Model
from train import STEP, peak_rate, run

WIDTHS = [4, 8, 16, 32, 64, 128, 256, 384]
STEPS = [61, 122, 244, 488, 977, 1953, 3906, 7812, 15625]
SHAPES = [(1, 90), (4, 46), (8, 32), (8, 128)]   # (layers, width)
SHAPE_STEPS = 3906
DICE_STEPS = 977   # these runs are repeated with other dice
# work = 6 x size x letters. The middle budget is the work of
# width 64 reading 977 steps of text. The others: 4 times less, more.
BUDGET = 6 * Model(64, 2).size() * 977 * STEP
BUDGETS = [BUDGET // 4, BUDGET, BUDGET * 4]
OUT = "files/runs.jsonl"


def finished():
    if not os.path.exists(OUT):
        return []
    lines = open(OUT).read().split("\n")[:-1]    # whole lines only
    return [json.loads(line) for line in lines]


def tries(width, layers, steps, seed):
    """Which step sizes to try for this run.

    Short runs are cheap, so they try four. Middle runs try three.
    The two longest runs reuse the winner of the 3,906-step run.
    See also one_more below.
    """
    base = peak_rate(width)
    if steps <= 977:
        return [base / 4, base / 2, base, base * 2]
    if steps <= 3906:
        return [base / 2, base, base * 2]
    same = [r for r in finished()
            if (r["width"], r["layers"], r["steps"], r["seed"])
            == (width, layers, 3906, seed)]
    return [min(same, key=lambda r: r["check"])["peak"]]


def one_more(width, layers, steps, seed):
    """If the best step size is the biggest or the smallest tried,
    the true best may lie further out: go one step past that end."""
    same = [r for r in finished()
            if (r["width"], r["layers"], r["steps"], r["seed"])
            == (width, layers, steps, seed)]
    if steps > 3906 or not 3 <= len(same) < 6:
        return None
    peaks = sorted(r["peak"] for r in same)
    best = min(same, key=lambda r: r["check"])["peak"]
    if best == peaks[-1]:
        return best * 2
    if best == peaks[0]:
        return best / 2
    return None


def jobs(widths):
    todo = []
    for width in widths:
        if width not in WIDTHS:
            continue    # a width used only by the shape test
        for steps in STEPS:
            todo.append((width, 2, steps, 0))
    for layers, width in SHAPES:
        if width in widths:
            todo.append((width, layers, SHAPE_STEPS, 0))
    # two layers first, short before long, small before big
    return sorted(todo, key=lambda j: (j[1] != 2, j[2], j[0]))


def budget_jobs():
    """For each budget and width: how many steps does the work buy?

    Skips runs that are too short to learn, need more text than
    we have, or are already in the big table.
    """
    todo = []
    for budget in BUDGETS:
        for width in WIDTHS:
            size = Model(width, 2).size()
            steps = round(budget / (6 * size * STEP))
            if 61 <= steps <= 21000 and steps not in STEPS:
                todo.append((width, steps, budget))
    return sorted(todo, key=lambda j: j[1])


def nearest_peak(width, steps):
    """The winning step size of the table run closest in length."""
    same = [r for r in finished()
            if (r["width"], r["layers"], r["seed"]) == (width, 2, 0)
            and r["steps"] in STEPS]
    lengths = {r["steps"] for r in same}
    near = min(lengths, key=lambda n: abs(math.log(n / steps)))
    return min((r for r in same if r["steps"] == near),
               key=lambda r: r["check"])["peak"]


def save(result, device, note=""):
    result["sample"] = write(result.pop("model"), device)
    with open(OUT, "a") as out:
        out.write(json.dumps(result) + "\n")
    print(f"w{result['width']} L{result['layers']} "
          f"steps={result['steps']} seed={result['seed']} "
          f"peak={result['peak']:.5f} size={result['size']} "
          f"loss={result['loss']:.4f} {result['seconds']}s{note}",
          flush=True)


@torch.no_grad()
def write(model, device, count=160, seed=0):
    """Let the model write: roll its loaded 27-sided die each time."""
    dice = torch.Generator().manual_seed(seed)
    text = [20, 8, 5, 0]    # "the "
    for _ in range(count):
        seen = torch.tensor([text[-64:]], device=device)
        chances = torch.softmax(model(seen)[0, -1].cpu(), dim=0)
        text.append(int(torch.multinomial(chances, 1, generator=dice)))
    return "".join(" abcdefghijklmnopqrstuvwxyz"[i] for i in text)


if __name__ == "__main__":
    device = sys.argv[1]
    os.makedirs("files", exist_ok=True)
    if sys.argv[2] == "budget":
        for width, steps, budget in budget_jobs():
            if any((r["width"], r["steps"]) == (width, steps)
                   for r in finished()):
                continue
            result = run(width, 2, steps * STEP, 0, device,
                         nearest_peak(width, steps), keep=True)
            result["budget"] = budget
            save(result, device, " budget")
        sys.exit()
    if sys.argv[2] == "dice":
        for seed in (1, 2):
            for width in WIDTHS:
                if any((r["width"], r["steps"], r["seed"])
                       == (width, DICE_STEPS, seed) for r in finished()):
                    continue
                peak = nearest_peak(width, DICE_STEPS)
                save(run(width, 2, DICE_STEPS * STEP, seed, device,
                         peak, keep=True), device, " dice")
        sys.exit()
    widths = [int(w) for w in sys.argv[2:]]
    for width, layers, steps, seed in jobs(widths):
        have = {r["peak"] for r in finished()
                if (r["width"], r["layers"], r["steps"], r["seed"])
                == (width, layers, steps, seed)}
        for peak in tries(width, layers, steps, seed):
            if any(abs(peak / h - 1) < 1e-6 for h in have):
                continue
            save(run(width, layers, steps * STEP, seed, device,
                     peak, keep=True), device)
        while one_more(width, layers, steps, seed):
            peak = one_more(width, layers, steps, seed)
            save(run(width, layers, steps * STEP, seed, device,
                     peak, keep=True), device, " one more")
