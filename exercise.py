"""Exercise. Step 4 tested the rule on models that read 4 million
letters. Does the same test pass when every model reads 16 million?

    python exercise.py

Your part: the three marked lines in `answer`. Find the rule from
the 5 smallest models, forecast the 3 biggest, and compare. The
answer of the video is printed at the end.
"""
from laws import column, fit_floor, load
from train import STEP

LETTERS = 3906 * STEP       # 16 million letters
FIT = 5                     # the rule is found from the 5 smallest


def answer(runs):
    rows = column(runs, LETTERS)
    sizes = [size for size, _ in rows]
    scores = [score for _, score in rows]
    # your part starts here
    floor, power, scale = fit_floor(sizes[:FIT], scores[:FIT])
    said = [floor + scale * size ** power for size in sizes[FIT:]]
    misses = [s - score for s, score in zip(said, scores[FIT:])]
    # and ends here
    return floor, sizes[FIT:], said, scores[FIT:], misses


if __name__ == "__main__":
    floor, sizes, said, scores, misses = answer(load())
    print("every model reads 16 million letters")
    print(f"floor found from the 5 smallest models: {floor:.2f}")
    print("\nsize (numbers)   forecast   measured   miss")
    for size, s, score, miss in zip(sizes, said, scores, misses):
        print(f"{size:>14,}   {s:>8.3f}   {score:>8.3f}   {miss:>+6.3f}")
    print(f"\nbiggest miss: {max(abs(m) for m in misses):.3f} bits")
