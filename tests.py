"""Checks of the claims of the video. python tests.py

Nothing here trains a model. The checks read files/runs.jsonl, the
table of finished runs, and repeat the sums of the video.
"""
import math

from exercise import answer
from laws import (best_split, column, compute, fit_floor, fit_law,
                  formula, load, same_work, score, table)
from model import LETTERS, WINDOW, Model
from step5_text import floors
from step6_formula import misses, points
from step7_budget import BUDGET, BUDGETS
from step8_schedule import on_the_way, planned
from step9_checks import dice, shapes
from sweep import WIDTHS
from train import BATCH, STEP, rate_at

done = []


def check(text, ok):
    done.append(bool(ok))
    print(("ok    " if ok else "WRONG ") + text)


def close(a, b, by=0.006):
    return abs(a - b) < by


runs = load()
M4, M16, M64 = 977 * STEP, 3906 * STEP, 15625 * STEP

# the game and the models
check("27 letters, a window of 64 letters, 4,096 letters per step",
      (LETTERS, WINDOW, BATCH, STEP) == (27, 64, 64, 4096))
blind = math.log2(27)
check(f"a guesser that knows nothing scores {blind:.2f} bits",
      close(blind, 4.755))
sizes = [Model(width, 2).size() for width in WIDTHS]
check("the eight sizes: " + ", ".join(f"{s:,}" for s in sizes),
      sizes == [496, 1760, 6592, 25472, 100096, 396800, 1580032,
                3549696])
check(f"the biggest model is {sizes[-1] / sizes[0]:,.0f} times the"
      " smallest", close(sizes[-1] / sizes[0], 7157, 1))
check("977 steps read 4 million letters, 15,625 steps read 64"
      " million", (M4, M64) == (4001792, 64000000))

# the plan of the step size
rates = [rate_at(step, 1000, 1.0) for step in range(1000)]
check("the step size rises for a tenth of the run, peaks, and ends"
      f" at {rates[-1]:.2f} of the peak",
      rates.index(max(rates)) in (99, 100) and close(max(rates), 1, 0.02)
      and close(rates[-1], 0.1, 0.01)
      and all(a >= b for a, b in zip(rates[100:], rates[101:])))
check("a quarter of the way through a long run, the moves are still"
      f" {rate_at(250, 1000, 1.0):.2f} of the peak",
      close(rate_at(250, 1000, 1.0), 0.94, 0.01))

# bigger models, the same 4 million letters
col = [loss for _, loss in column(runs, M4)]
check("the eight scores after 4 million letters: "
      + ", ".join(f"{c:.2f}" for c in col),
      all(close(c, want) for c, want in zip(col, [
          3.612, 3.211, 2.869, 2.611, 2.439, 2.295, 2.205, 2.139])))
check("every model beats the one before it",
      all(a > b for a, b in zip(col, col[1:])))
check(f"the first step gains {col[0] - col[1]:.2f} bits, the sixth"
      f" gains {col[5] - col[6]:.2f}",
      close(col[0] - col[1], 0.40, 0.01) and col[5] - col[6] < 0.1)
ratios = [b / a for a, b in zip(col, col[1:])]
check("the score itself shrinks by a smaller share each step: "
      + ", ".join(f"{r:.2f}" for r in ratios[:4]),
      [round(r, 2) for r in ratios[:4]] == [0.89, 0.89, 0.91, 0.93])

# the floor and the power law, from the 5 smallest models
floor, power, scale = fit_floor(sizes[:5], col[:5])
gaps = [c - floor for c in col]
check(f"the floor under the 5 smallest models is {floor:.2f}",
      close(floor, 1.89))
check(f"the first two gaps are {gaps[0]:.2f} and {gaps[1]:.2f} bits",
      close(gaps[0], 1.72, 0.01) and close(gaps[1], 1.32, 0.01))
steps = [b / a for a, b in zip(gaps[:5], gaps[1:5])]
check("each step leaves this share of the gap: "
      + ", ".join(f"{s:.2f}" for s in steps),
      close(steps[0], 0.77, 0.01)
      and all(0.735 < s < 0.765 for s in steps[1:]))
check(f"4 times the size leaves {4 ** power:.2f} of the gap, 10 times"
      f" leaves {10 ** power:.2f}",
      close(4 ** power, 0.74, 0.01) and close(10 ** power, 0.61, 0.01))
check(f"to halve the gap: {0.5 ** (1 / power):.0f} times the size",
      close(0.5 ** (1 / power), 24, 0.5))

# the forecast for the 3 biggest models
said = [floor + scale * size ** power for size in sizes[5:]]
check("forecasts " + ", ".join(f"{s:.2f}" for s in said)
      + "   measured " + ", ".join(f"{c:.2f}" for c in col[5:]),
      all(close(s, want) for s, want in zip(said, [2.292, 2.188,
                                                   2.140])))
worst = max(abs(s - c) for s, c in zip(said, col[5:]))
check(f"the biggest miss is {worst:.3f} bits", worst < 0.025)
check(f"the biggest model is {sizes[-1] / sizes[4]:.0f} times larger"
      " than any model the line was built from",
      close(sizes[-1] / sizes[4], 35.5, 0.1))

# more text
big = [score(runs, 384, 2, letters) for letters in (M4, M16, M64)]
check("the biggest model with 4, 16 and 64 million letters: "
      + ", ".join(f"{b:.2f}" for b in big),
      all(close(b, want) for b, want in zip(big, [2.139, 1.920,
                                                  1.812])))
check("with 64 million letters it is under the first floor",
      big[2] < floor)
found = {row[0]: row[1] for row in floors(runs)}
check("the floors under all eight models: "
      + ", ".join(f"{found[t]:.2f}" for t in (M4, M16, M64)),
      close(found[M4], 1.905, 0.011) and close(found[M16], 1.685, 0.011)
      and close(found[M64], 1.585, 0.011))
few = 61 * STEP
gain = [score(runs, 64, 2, t) - score(runs, 384, 2, t)
        for t in (few, M64)]
check(f"35 times the size gains {gain[0]:.2f} bits with a quarter"
      f" of a million letters, {gain[1]:.2f} with 64 million",
      close(gain[0], 0.08, 0.01) and close(gain[1], 0.26, 0.01))

# one formula for size and text
measured = points(runs)
law = fit_law(measured)
off = misses(law, measured)
usual = (sum(m * m for m in off) / len(off)) ** 0.5
check(f"the formula is fitted to {len(measured)} runs with 4 million"
      " letters or more", len(measured) == 40)
check(f"its floor is {law[0]:.2f}, under every measured score",
      law[0] < min(measured.values()))
check(f"4 times the size leaves {4 ** -law[2]:.2f} of the model part,"
      f" 4 times the text leaves {4 ** -law[4]:.2f} of the text part",
      0.6 < 4 ** -law[2] < 0.8 and 0.4 < 4 ** -law[4] < 0.6)
check(f"usual miss {usual:.3f} bits, biggest {max(map(abs, off)):.3f}",
      usual < 0.05 and max(map(abs, off)) < 0.15)

# a fixed amount of work
check(f"the middle budget is {BUDGET / 1e12:.2f} million million"
      " calculations", BUDGET == compute(100096, M4)
      and close(BUDGET / 1e12, 2.40, 0.01))
rows = same_work(runs, BUDGET)
losses = [r["loss"] for r in rows]
check("five models for the same work score "
      + ", ".join(f"{x:.2f}" for x in losses),
      [r["size"] for r in rows] == sizes[2:7]
      and all(close(x, want) for x, want in zip(losses, [
          2.657, 2.388, 2.439, 2.835, 3.447])))
check("the best of the five has 25,472 numbers: not the smallest,"
      " not the biggest", losses.index(min(losses)) == 1)
best = [min(same_work(runs, b), key=lambda r: r["loss"])["size"]
        for b in BUDGETS]
check("the best measured size for a quarter, one and four times the"
      " budget: " + ", ".join(f"{b:,}" for b in best),
      best == [25472, 25472, 100096])
splits = [best_split(b, law)[0] for b in BUDGETS]
check("the formula's best size grows with the budget: "
      + ", ".join(f"{s:,.0f}" for s in splits),
      splits[0] < splits[1] < splits[2])
share = law[4] / (law[2] + law[4])
check(f"for 10 times the work the formula grows the model"
      f" {10 ** share:.1f} times and the text {10 ** (1 - share):.1f}"
      " times", 10 ** share > 10 ** (1 - share))

# read on the way, or planned
way = on_the_way(runs, 64, 3904)
own = planned(runs, 64, 3906)[1]
check(f"100,096 numbers after 16 million letters: {own:.2f} planned,"
      f" {way:.2f} on the way to 64 million",
      close(own, 2.189) and close(way, 2.315))
check(f"on the way it is {way - own:.2f} bits worse",
      close(way - own, 0.126))

# the two checks
flat, deep = shapes(runs)
check("100,000 numbers in 1, 2, 4 and 8 layers: "
      + ", ".join(f"{s[3]:.2f}" for s in flat),
      max(s[3] for s in flat) - min(s[3] for s in flat) < 0.07)
check(f"1.58 million numbers in 2 and 8 layers: {deep[0][3]:.2f} and"
      f" {deep[1][3]:.2f}", abs(deep[0][3] - deep[1][3]) < 0.01)
gaps = [max(s) - min(s) for _, s in dice(runs)]
check(f"other dice move the smallest model by {gaps[0]:.2f} bits,"
      f" every other model by at most {max(gaps[1:]):.2f}",
      close(gaps[0], 0.110) and max(gaps[1:]) < 0.045)

# the exercise
_, _, said, real, miss = answer(runs)
check("the exercise, 16 million letters: forecasts "
      + ", ".join(f"{s:.2f}" for s in said) + ", biggest miss"
      f" {max(map(abs, miss)):.2f} bits",
      close(max(map(abs, miss)), 0.139))

print(f"\n{sum(done)} of {len(done)} checks ok")
raise SystemExit(0 if all(done) else 1)
