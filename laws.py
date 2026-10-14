"""Read the finished runs and find the laws in them.

Nothing here trains a model. Every function works on the table of
results in files/runs.jsonl: (size, letters read) -> test loss.
"""
import json
import math

import numpy as np

RUNS = "files/runs.jsonl"


def load(path=RUNS):
    """Every finished run, with its losses turned into bits.

    The training code measures in "nats". One bit is 0.693 nats,
    so dividing by 0.693 gives bits: the number of halvings.
    """
    runs = [json.loads(line) for line in open(path)]
    for r in runs:
        r["loss"] /= math.log(2)
        r["check"] /= math.log(2)
        for mark in r["curve"]:
            mark["test"] /= math.log(2)
            mark["train"] /= math.log(2)
    return runs


def best_runs(runs):
    """Of the step sizes tried for one run, keep the best one.

    Best is judged on the check letters, never on the test letters.
    """
    best = {}
    for r in runs:
        key = (r["width"], r["layers"], r["letters"], r["seed"])
        if key not in best or r["check"] < best[key]["check"]:
            best[key] = r
    return list(best.values())


def table(runs, layers=2, seed=0):
    """The big table: {(size, letters): test loss}."""
    found = {}
    for r in best_runs(runs):
        if (r["layers"], r["seed"]) == (layers, seed) \
                and "budget" not in r:
            found[(r["size"], r["letters"])] = r["loss"]
    return dict(sorted(found.items()))


def score(runs, width, layers, letters, seed=0):
    """The test loss of one run, by its shape and its text."""
    for r in best_runs(runs):
        if (r["width"], r["layers"], r["letters"], r["seed"]) \
                == (width, layers, letters, seed):
            return r["loss"]
    return None


def column(runs, letters):
    """One column of the table: [(size, loss)] for one amount of
    text, smallest model first."""
    return [(size, loss) for (size, text), loss in table(runs).items()
            if text == letters]


def fit_line(xs, ys):
    """The straight line closest to the points: (slope, start)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    top = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    bottom = sum((x - mx) ** 2 for x in xs)
    slope = top / bottom
    return slope, my - slope * mx


def fit_power(xs, ys):
    """A straight line on log paper: y = scale * x ** power."""
    slope, start = fit_line(
        [math.log(x) for x in xs], [math.log(y) for y in ys])
    return slope, math.exp(start)


def fit_floor(xs, ys):
    """y = floor + scale * x ** power. Returns (floor, power, scale).

    Try every floor from 0 up to just under the lowest point, in
    steps of 0.005. Keep the floor whose line misses the least.
    """
    best = None
    floor = 0.0
    while floor < min(ys) - 0.005:
        power, scale = fit_power(xs, [y - floor for y in ys])
        miss = sum((floor + scale * x ** power - y) ** 2
                   for x, y in zip(xs, ys))
        if best is None or miss < best[0]:
            best = (miss, floor, power, scale)
        floor += 0.005
    return best[1:]


def formula(size, letters, law):
    """Loss = floor + (model too small) + (too little text)."""
    floor, a, alpha, b, beta = law
    return floor + a / size ** alpha + b / letters ** beta


def fit_law(points):
    """Find the five numbers of the formula from measured points.

    points: {(size, letters): loss}. For one pair of powers
    (alpha, beta) the other three numbers have an exact best
    answer, so we try many pairs and keep the best.
    """
    keys = list(points)
    size = np.array([k[0] for k in keys], dtype=float)
    text = np.array([k[1] for k in keys], dtype=float)
    loss = np.array([points[k] for k in keys])
    best = None
    grid = np.arange(0.05, 1.5001, 0.01)
    for alpha in grid:
        for beta in grid:
            parts = np.stack(
                [np.ones_like(size), size ** -alpha, text ** -beta], 1)
            answer, *_ = np.linalg.lstsq(parts, loss, rcond=None)
            miss = float(np.mean((parts @ answer - loss) ** 2))
            if min(answer) > 0 and (best is None or miss < best[0]):
                best = (miss, (float(answer[0]), float(answer[1]),
                               float(alpha), float(answer[2]),
                               float(beta)))
    return best[1]


def compute(size, letters):
    """Work to train: every letter touches every number 6 times."""
    return 6 * size * letters


def same_work(runs, budget, within=0.005):
    """The runs that cost this much work, smallest model first.

    One run for each size: the one closest to the budget.
    """
    found = {}
    for r in best_runs(runs):
        off = abs(compute(r["size"], r["letters"]) / budget - 1)
        if (r["layers"], r["seed"]) != (2, 0) or off >= within:
            continue
        if r["size"] not in found or off < found[r["size"]][0]:
            found[r["size"]] = (off, r)
    return [r for _, r in sorted(found.values(),
                                 key=lambda pair: pair[1]["size"])]


def best_split(budget, law, steps=2000):
    """For a fixed amount of work, which size gives the lowest loss?"""
    best = None
    for i in range(steps + 1):
        size = 10 ** (2 + 6 * i / steps)
        letters = budget / (6 * size)
        loss = formula(size, letters, law)
        if best is None or loss < best[2]:
            best = (size, letters, loss)
    return best
