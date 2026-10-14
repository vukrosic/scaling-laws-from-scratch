"""Step 8. Why two famous papers disagree: the training plan.

    python step8_schedule.py

In every run the step size falls along a plan made for the length
of that run (rate_at in train.py). So there are two ways to get
the score "after 4 million letters":

  planned: a run planned to stop at 4 million letters
  on the way: a run planned for 64 million letters, read off when
              it has seen 4 million. Its steps are still big.

No model is trained here. The long runs wrote down their score
on the way (the "curve" of each run in files/runs.jsonl).

At the end the formula of step 6 is fitted twice, once to the
planned scores and once to the scores read off on the way, to see
whether the best split of the work changes.
"""
from laws import best_runs, fit_law, load
from step6_formula import points
from train import STEP

LONG = 15625 * STEP
MARKS = [(976, 977), (3904, 3906)]      # mark on the way, planned run
ALL = [(61, 61), (122, 122), (244, 244), (488, 488), (976, 977),
       (1952, 1953), (3904, 3906), (7808, 7812)]
MARK_OF = {own: mark for mark, own in ALL}


def on_the_way(runs, width, step):
    """The long run of this width, read off after `step` steps."""
    for r in best_runs(runs):
        if (r["width"], r["layers"], r["letters"], r["seed"]) \
                == (width, 2, LONG, 0):
            for mark in r["curve"]:
                if mark["letters"] == step * STEP:
                    return mark["test"]


def planned(runs, width, steps):
    for r in best_runs(runs):
        if (r["width"], r["layers"], r["steps"], r["seed"]) \
                == (width, 2, steps, 0) and "budget" not in r:
            return r["size"], r["loss"]


def way_points(runs):
    """Like points() of step 6, but every score is read off the
    long run on the way. The last column is the long run's end."""
    found = {}
    for size, letters in points(runs):
        width = next(r["width"] for r in runs if r["size"] == size
                     and r["layers"] == 2)
        if letters == LONG:
            mark = planned(runs, width, 15625)[1]
        else:
            mark = on_the_way(runs, width, MARK_OF[letters // STEP])
        if mark is not None:
            found[size, letters] = mark
    return found


def size_share(law):
    """Of 4 times the work, the part that goes to model size.

    At the best split, size grows as work ^ share."""
    alpha, beta = law[2], law[4]
    return beta / (alpha + beta)


if __name__ == "__main__":
    runs = load()
    widths = sorted({r["width"] for r in runs if r["layers"] == 2
                     and r["letters"] == LONG})
    for mark, steps in MARKS:
        print(f"score after {steps * STEP / 1e6:.0f} million letters")
        print("    numbers   on the way   planned   difference")
        for width in widths:
            size, own = planned(runs, width, steps)
            mid = on_the_way(runs, width, mark)
            print(f"  {size:>9,}   {mid:>10.3f}   {own:>7.3f}"
                  f"   {mid - own:>+10.3f}")
        print()

    print("best split of 4 times the work, from the formula")
    print("                  model      text")
    for name, measured in (("planned", points(runs)),
                           ("on the way", way_points(runs))):
        share = size_share(fit_law(measured))
        print(f"  {name:<12}  x {4 ** share:.2f}    x {4 ** (1 - share):.2f}"
              f"   ({len(measured)} scores)")
