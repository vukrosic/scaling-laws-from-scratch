"""Step 4. The test of the rule: forecast models it never saw.

    python step4_forecast.py

The floor and the line come from the 5 smallest models. The 3
biggest were left out. Now the line is asked for their scores.
"""
from laws import column, fit_floor, fit_power, load
from train import STEP

LETTERS = 977 * STEP
FIT = 5

if __name__ == "__main__":
    rows = column(load(), LETTERS)
    sizes = [size for size, _ in rows]
    scores = [score for _, score in rows]
    floor, power, scale = fit_floor(sizes[:FIT], scores[:FIT])

    print(f"built from sizes up to {sizes[FIT - 1]:,}")
    print("\nsize (numbers)   forecast   measured   miss")
    worst = 0
    for size, score in rows[FIT:]:
        forecast = floor + scale * size ** power
        worst = max(worst, abs(forecast - score))
        print(f"{size:>14,}   {forecast:>8.3f}   {score:>8.3f}"
              f"   {forecast - score:>+6.3f}")
    print(f"\nbiggest miss: {worst:.3f} bits")
    reach = sizes[-1] / sizes[FIT - 1]
    print(f"the biggest model is {reach:.1f} times larger than any"
          " model the line was built from")

    # what would it cost to halve the gap?
    print(f"\nto halve the gap: {0.5 ** (1 / power):.0f} times the size")

    # the line with no floor, for comparison
    power0, scale0 = fit_power(sizes[:FIT], scores[:FIT])
    print("\nwith no floor (a plain straight line on log paper):")
    for size, score in rows[FIT:]:
        print(f"{size:>14,}   {scale0 * size ** power0:>8.2f}"
              f"   {score:>8.2f}")
