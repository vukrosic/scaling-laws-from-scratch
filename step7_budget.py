"""Step 7. A fixed amount of work: bigger model, or more text?

    python step7_budget.py

Work = 6 x numbers x letters. Every run in one block below cost
the same work. Only the split changes: a small model that reads a
lot, or a big model that reads little. Most of these runs were not
used to find the formula of step 6, so its forecast for them is a
second test.
"""
from laws import best_split, compute, fit_law, formula, load, same_work
from step6_formula import points
from train import STEP

BUDGET = compute(100096, 977 * STEP)    # the middle amount of work
BUDGETS = [BUDGET // 4, BUDGET, BUDGET * 4]

if __name__ == "__main__":
    runs = load()
    law = fit_law(points(runs))
    floor, a, alpha, b, beta = law

    for budget in BUDGETS:
        print(f"work: {budget / 1e12:.2f} million million")
        print("    numbers       letters   measured   formula")
        rows = same_work(runs, budget)
        low = min(rows, key=lambda r: r["loss"])
        for r in rows:
            said = formula(r["size"], r["letters"], law)
            mark = "   <- lowest" if r is low else ""
            print(f"  {r['size']:>9,}   {r['letters']:>11,}"
                  f"   {r['loss']:>8.3f}   {said:>7.3f}{mark}")
        size, letters, loss = best_split(budget, law)
        print(f"  the formula's best split: {size:,.0f} numbers,"
              f" {letters / 1e6:.1f}M letters, score {loss:.2f}\n")

    # the rule behind the best split
    share = beta / (alpha + beta)
    print("when the work grows, the formula's best split grows as")
    print(f"  model = work ^ {share:.2f}     text = work ^"
          f" {1 - share:.2f}")
    for times in (4, 10):
        print(f"  {times:>2} times the work: model x"
              f" {times ** share:.1f}, text x"
              f" {times ** (1 - share):.1f}")
