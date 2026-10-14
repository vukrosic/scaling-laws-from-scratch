"""Step 9. Two checks: the shape of the model, and the dice.

    python step9_checks.py

Shape: the same count of numbers, built flat and wide or deep and
narrow. Dice: the same run again, with other random start numbers
and another order of the text.
"""
from laws import load, score
from model import Model
from train import STEP

SHAPE_TEXT = 3906 * STEP    # 16 million letters
SHAPES = [[(1, 90), (2, 64), (4, 46), (8, 32)],     # (layers, width)
          [(2, 256), (8, 128)]]
DICE_TEXT = 977 * STEP      # 4 million letters
WIDTHS = [4, 8, 16, 32, 64, 128, 256, 384]


def shapes(runs):
    return [[(layers, width, Model(width, layers).size(),
              score(runs, width, layers, SHAPE_TEXT))
             for layers, width in group] for group in SHAPES]


def dice(runs):
    """For each size: its scores with dice 0, 1 and 2."""
    return [(Model(width, 2).size(),
             [score(runs, width, 2, DICE_TEXT, seed)
              for seed in (0, 1, 2)]) for width in WIDTHS]


if __name__ == "__main__":
    runs = load()
    print("shape, with 16 million letters")
    for group in shapes(runs):
        print("  layers   width     numbers   score")
        for layers, width, size, bits in group:
            print(f"  {layers:>6}   {width:>5}   {size:>9,}"
                  f"   {bits:.3f}")
        print()

    print("dice, with 4 million letters")
    print("    numbers   dice 0   dice 1   dice 2   widest gap")
    worst = 0
    for size, scores in dice(runs):
        gap = max(scores) - min(scores)
        worst = max(worst, gap)
        print(f"  {size:>9,}" + "".join(f"   {s:.3f}" for s in scores)
              + f"   {gap:>10.3f}")
    print(f"\nthe dice move a score by at most {worst:.3f} bits")
