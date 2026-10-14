"""Step 6. One formula for both: model size and amount of text.

    python step6_formula.py

score = floor + (part for a model too small)
              + (part for too little text)

Each part is a power law, like the gap of step 3. The five numbers
of the formula are fitted to every run with 4 million letters or
more. With less text a model has barely started to learn, and the
formula does not describe that start.
"""
from laws import fit_law, formula, load, table
from train import STEP

ENOUGH = 977 * STEP         # 4 million letters


def points(runs):
    return {key: loss for key, loss in table(runs).items()
            if key[1] >= ENOUGH}


def misses(law, measured):
    return [formula(size, text, law) - loss
            for (size, text), loss in measured.items()]


if __name__ == "__main__":
    measured = points(load())
    law = fit_law(measured)
    floor, a, alpha, b, beta = law
    print(f"fitted to {len(measured)} runs\n")
    print(f"score = {floor:.2f} + {a:.2f} / size ^ {alpha:.2f}"
          f" + {b:.0f} / letters ^ {beta:.2f}\n")
    print(f"floor of the formula: {floor:.2f} bits per letter")
    print(f"4 times the size:    model part x {4 ** -alpha:.2f}")
    print(f"4 times the letters: text part  x {4 ** -beta:.2f}")

    off = misses(law, measured)
    mean = (sum(m * m for m in off) / len(off)) ** 0.5
    print(f"\nusual miss: {mean:.3f} bits"
          f"   biggest miss: {max(abs(m) for m in off):.3f} bits")

    print("\n   size       letters   model part  text part"
          "  formula  measured")
    for (size, text), loss in measured.items():
        if size in (1760, 100096, 3549696):
            print(f"{size:>9,}  {text:>11,}   {a / size ** alpha:>8.3f}"
                  f"   {b / text ** beta:>8.3f}"
                  f"   {formula(size, text, law):>6.3f}   {loss:>7.3f}")
