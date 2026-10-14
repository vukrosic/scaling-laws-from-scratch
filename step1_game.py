"""Step 1. The game: guess the next letter. The score: bits.

    python step1_game.py

No model yet. Two guessers that need no training show what the
score means: one that knows nothing, one that knows how often
each letter appears.
"""
import math

import torch

from train import TEST_FROM, text

NAMES = " abcdefghijklmnopqrstuvwxyz"


def show(numbers):
    return "".join(NAMES[i] for i in numbers)


def bits(chance):
    """How many times 1 must be halved to reach this chance."""
    return math.log2(1 / chance)


if __name__ == "__main__":
    book = text()
    print("letters in the text:", f"{len(book):,}")
    print("it starts:", repr(show(book[:66])))

    print("\nscore of one guess = halvings from 1 down to the chance")
    for chance in (1, 1 / 2, 1 / 4, 1 / 8, 1 / 27):
        print(f"  chance {chance:.3f} -> {bits(chance):.2f} bits")

    # guesser 1 knows nothing: every one of 27 symbols gets 1/27
    print(f"\nknows nothing: {bits(1 / 27):.2f} bits per letter")

    # guesser 2 counts how often each symbol appears in the
    # training letters, then always answers with those 27 chances
    count = torch.bincount(book[:TEST_FROM], minlength=27).double()
    chance = count / count.sum()
    for i in chance.argsort(descending=True)[:4].tolist() + [17]:
        print(f"  {NAMES[i]!r}: {100 * chance[i]:.2f} % of letters,"
              f" {bits(chance[i]):.2f} bits when it comes")
    test = book[TEST_FROM:]
    score = -torch.log2(chance[test.long()]).mean().item()
    print(f"knows the 27 counts: {score:.2f} bits per letter")
