"""Step 3. Eight sizes, the same text. Is there a rule?

    python step3_size.py

Every model read the same 4 million letters once. Only the size
changes. The first rule we try fails. The second one holds.
"""
from laws import column, fit_floor, load
from train import STEP

LETTERS = 977 * STEP        # 4 million letters
FIT = 5                     # the rule is found from the 5 smallest

if __name__ == "__main__":
    rows = column(load(), LETTERS)
    sizes = [size for size, _ in rows]
    scores = [score for _, score in rows]

    print("size (numbers)   score (bits per letter)   gain")
    for i, (size, score) in enumerate(rows):
        gain = f"{scores[i - 1] - score:.2f}" if i else ""
        print(f"{size:>14,}   {score:>10.3f}   {gain:>16}")

    print("\nrule 1: every step takes the same share off the score")
    for i in range(1, FIT):
        print(f"  step {i}: score x {scores[i] / scores[i - 1]:.3f}")
    print("  the share shrinks at every step: rule 1 fails")

    floor, power, scale = fit_floor(sizes[:FIT], scores[:FIT])
    print("\nrule 2: the scores settle above a floor")
    print(f"  floor that fits the {FIT} smallest: {floor:.2f} bits")
    gaps = [score - floor for score in scores]
    for i in range(FIT):
        times = f"x {gaps[i] / gaps[i - 1]:.2f}" if i else ""
        print(f"  {sizes[i]:>9,}   gap {gaps[i]:.2f}   {times}")
    print(f"\n  gap = {scale:.2f} x size ^ {power:.3f}")
    print(f"  4 times the size:  gap x {4 ** power:.2f}")
    print(f"  10 times the size: gap x {10 ** power:.2f}")
