"""Step 5. Change the other thing: the amount of text.

    python step5_text.py

Until now every model read the same 4 million letters. Here the
same eight sizes read less text and more text. The floor of
step 3 was not a fact about language. It moves with the text.
"""
from laws import column, fit_floor, load, table
from train import STEP

TEXTS = [61, 244, 977, 3906, 15625]     # steps of 4,096 letters
FLOORS = [488, 977, 3906, 15625]        # columns with a clear floor


def name(letters):
    return f"{letters / 1e6:.2f}M"


def floors(runs):
    """For each amount of text: the floor under all eight sizes."""
    found = []
    for steps in FLOORS:
        rows = column(runs, steps * STEP)
        floor, power, scale = fit_floor([size for size, _ in rows],
                                        [loss for _, loss in rows])
        found.append((steps * STEP, floor, power, scale))
    return found


if __name__ == "__main__":
    runs = load()
    scores = table(runs)
    sizes = sorted({size for size, _ in scores})

    print("score in bits per letter. Columns: letters read\n")
    print(" " * 14 + "".join(f"{name(s * STEP):>9}" for s in TEXTS))
    for size in sizes:
        row = "".join(f"{scores[(size, s * STEP)]:>9.3f}"
                      for s in TEXTS)
        print(f"{size:>14,}{row}")

    print("\nthe floor under all eight sizes, for each amount of text")
    for letters, floor, power, _ in floors(runs):
        print(f"  {name(letters):>7} letters: floor {floor:.2f} bits"
              f"   (4 times the size: gap x {4 ** power:.2f})")

    small, big = sizes[4], sizes[-1]
    print(f"\nfrom {small:,} numbers to {big:,} numbers"
          f" ({big / small:.0f} times bigger):")
    for steps in (61, 15625):
        gain = scores[(small, steps * STEP)] - scores[(big, steps * STEP)]
        print(f"  with {name(steps * STEP)} letters the score improves"
              f" by {gain:.2f} bits")
