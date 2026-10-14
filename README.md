# Scaling laws from scratch

Code for the video **Why Bigger AI Models Keep Winning (Scaling Laws From Scratch)**.
A small GPT-style model reads English text one letter at a time and guesses the next letter. The same design is built in eight sizes, from 496 numbers to 3,549,696 numbers, and every size is trained on nine amounts of text, from a quarter of a million letters to 64 million. The 306 finished trainings (several step sizes per cell, runs of equal work, other dice, other shapes) are one file, and the best training of each cell makes one table. Nine small programs read that table and answer one question: make the model ten times bigger, how much better does it get, and can you know that before you build it?
About 1,400 lines of Python 3. The two papers the video follows: Kaplan and others, "Scaling Laws for Neural Language Models", 2020 (OpenAI), and Hoffmann and others, "Training Compute-Optimal Large Language Models", 2022 (DeepMind).

```bash
pip install numpy torch
curl -O http://mattmahoney.net/dc/text8.zip && unzip text8.zip   # 100 MB of text, see "The text"
sh run_all.sh         # nine steps, the tests and the exercise; expected output is in demo-output.txt
```

All of `run_all.sh` takes about 30 seconds on a laptop. Only step 2 trains a model (the smallest, 12 seconds). Every other number is read from `files/runs.jsonl` (231 KB), one line per finished training.
To train the table again, delete `files/runs.jsonl` and run the five commands at the top of `sweep.py`. On an Apple M4 laptop that took 5.7 hours. The sweep can be stopped and started again: it skips every training that already has a line.

## The text

`text8` is the first 100,000,000 characters of a cleaned English Wikipedia dump, made by Matt Mahoney: only the letters a to z and the space, 27 symbols, digits spelled out as words. It is not in this repository (100 MB). The page is http://mattmahoney.net/dc/textdata.html . Put the unzipped file `text8` next to `train.py`.
Without the file, steps 3 to 9, the tests and the exercise still run, because they only read the table. Steps 1 and 2 and the sweep need the text.

The last 5 million letters are the test. No model learns from them. The 5 million letters before those are used to pick the step size.

## The score

The model gives every one of the 27 symbols a chance. The score of one guess is the number of halvings from 1 down to the chance it gave the letter that really came: a chance of one half is 1 bit, one quarter is 2 bits. The score of a model is the average over about half a million test letters, in **bits per letter**. Lower is better. A guesser that knows nothing scores 4.75. A guesser that knows only how common each letter is scores 4.12.

```python
def loss_of(model, text):
    """Average surprise at the true next letter. Lower is better."""
    chances = model(text[:, :-1])
    return F.cross_entropy(
        chances.reshape(-1, LETTERS), text[:, 1:].reshape(-1))
```

`F.cross_entropy` counts in natural units; `laws.py` divides by ln 2 to get bits.

## Words

| In the video | In the papers | In the code |
| --- | --- | --- |
| letter | token (the papers use pieces of words) | one of 27 symbols: space is 0, a is 1, z is 26 |
| the box, the model | language model, transformer | `Model(width, layers)` in `model.py` |
| numbers in the box, size | parameters, N | `Model.size()`: all numbers without the two lookup tables, as the 2020 paper counts |
| amount of text | dataset size, D, training tokens | steps × 4,096 letters; every letter is read once |
| learning step | one update of the optimiser (Adam) | 64 windows of 64 letters, then every number moves a little |
| setting: how far the numbers move | learning rate | `peak_rate(width)`; `sweep.py` also tries half and double of it and keeps the best |
| the plan of the moves | learning rate schedule | `rate_at`: rises for a tenth of the run, then falls on a cosine to a tenth of the peak |
| score, bits per letter | test loss, cross-entropy | `loss` in `files/runs.jsonl`, divided by ln 2 |
| floor | irreducible loss, L∞ | `fit_floor`, `fit_law` |
| gap | reducible loss | score minus floor |
| straight line on log paper | power law | gap = scale × size ^ power |
| the rule | L(N, D) = E + A / N^α + B / D^β (the 2022 paper's form) | `formula(size, letters, law)` |
| work | compute, C, in FLOPs | `compute`: 6 × numbers × letters |
| budget | compute budget | `BUDGET`: the work of 100,096 numbers reading 4 million letters |
| best split | compute-optimal model size and data | `best_split(budget, law)` |
| read on the way | loss read off one long training curve | `on_the_way` in `step8_schedule.py` |
| other dice | another random seed | `seed` 1 and 2 |
| shape | depth against width | `SHAPES` in `sweep.py` |

## Files

| File | Idea | What the run shows |
| --- | --- | --- |
| `model.py` | The model | 90 lines. Letters in, 27 chances out. `width` and `layers` set its size. |
| `train.py` | One training | 113 lines. Reads the text once, 4,096 letters per step, and returns the test score. |
| `sweep.py` | Every size on every amount of text | Writes one line per finished training into `files/runs.jsonl`. |
| `laws.py` | Reading the table, fitting the rules | `fit_floor` (a floor and a straight line on log paper), `fit_law` (one rule for size and text), `best_split`. |
| `step1_game.py` | The game and its score | Knowing nothing: 4.75 bits per letter. Knowing the 27 counts: 4.12. |
| `step2_one_model.py` | The smallest model, trained here | 496 numbers. Before training 4.76 bits, after 4 million letters 3.61, with a line of its writing before and after. |
| `step3_size.py` | Eight sizes, the same 4 million letters | 3.61, 3.21, 2.87, 2.61, 2.44, 2.30, 2.21, 2.14. "The same share off the score every step" fails (× 0.89, 0.89, 0.91, 0.93). With a floor of 1.89, each step of 4 times the size leaves about 0.74 of the gap. |
| `step4_forecast.py` | A forecast for models the line never saw | Built from the 5 smallest. Forecasts 2.292, 2.188, 2.140. Measured 2.295, 2.205, 2.139. Biggest miss 0.017 bits, for a model 35 times larger than any the line was built from. |
| `step5_text.py` | More text | The whole table. The floor under all eight sizes is 1.90 with 4 million letters, 1.68 with 16 million, 1.58 with 64 million. From 100,096 to 3,549,696 numbers the score improves by 0.08 bits with a quarter of a million letters and by 0.26 with 64 million. |
| `step6_formula.py` | One rule for size and text | score = 1.52 + 8.28 / size ^ 0.24 + 1126 / letters ^ 0.53, fitted to 40 trainings. Usual miss 0.041 bits, biggest 0.133. |
| `step7_budget.py` | The same work, split five ways | Work of 2.40 million million calculations: 2.66, 2.39, 2.44, 2.83, 3.45. The best model is the second smallest. For 4 times the work the rule grows the model 2.6 times and the text 1.5 times. |
| `step8_schedule.py` | Read on the way, or planned | 100,096 numbers after 16 million letters: 2.19 with a plan made for 16 million, 2.32 when read on the way to 64 million. |
| `step9_checks.py` | Shape and dice | About 100,000 numbers in 1, 2, 4 and 8 layers: 2.24, 2.19, 2.18, 2.24. Other dice move the smallest model by 0.11 bits and every other model by at most 0.04. |
| `values.py` | Numbers for the pictures of the video | Writes `values.json`. Not part of `run_all.sh`. |
| `tests.py` | Checks of the claims | 39 checks. None of them trains a model. |
| `exercise.py` | The forecast, with 16 million letters | See below. |

## Reading the numbers

- **One kept training per cell.** Every score in the table is the best of up to four trainings that differ only in step size, all with dice 0. The dice check in step 9 repeats the 4-million-letter column with two more seeds: the widest gap is 0.11 bits for 496 numbers and 0.04 or less for the others. A difference smaller than that between two cells is not a result.
- **Every letter is read once.** A training with 4 million letters takes 977 steps, one with 64 million takes 15,625. No model sees a letter twice, so no model can learn its text by heart.
- **A plan for each length.** Every training has its own plan of step sizes, made for its own length. That is the 2022 paper's way. Step 8 shows what changes when scores are read on the way through one long training instead.
- **The step size.** The peak of the plan depends on the width (`peak_rate`). Trainings up to 4 million letters try four peaks, trainings up to 16 million try three, and the two longest reuse the winner of the 16-million run. The best is chosen on a piece of text that is not the test (`check` in `files/runs.jsonl`).
- **The floor in step 3 is not a limit of the models.** It is 1.89 for 4 million letters. The biggest model with 64 million letters scores 1.81, under it. Step 5 shows the floor moving with the amount of text.
- **Which trainings the rule is fitted to.** The 40 cells with 4 million letters or more. With less text the models are far from the rule and are left out.
- **The rule's own floor, 1.52, is a guess.** It is where the fitted curve levels off, beyond every measured score. The lowest measured score is 1.81.
- **Work.** 6 × numbers × letters, the count both papers use for one pass forward and back. Trainings of "equal work" are within half a percent of each other.
- **The measured best size and the rule's best size differ.** Our sizes are four times apart, so the measured best of a budget can only be one of five sizes. The rule's best split lies between them (58,749 numbers for the middle budget).

## What this is not

- **Not a reproduction of the papers' numbers.** Kaplan and others, 2020: for 10 times the work, a model about 5.5 times bigger and 1.8 times the text. Hoffmann and others, 2022: about 3 times each. This toy gives 4.9 and 2.0, with letters, one text, models under 4 million numbers and one training per cell.
- **Not a test of which paper is right.** Step 8 measures one cause the 2022 paper names (scores read on the way look worse than planned ones). On these runs it moves a score by up to 0.13 bits at 16 million letters. The best split the rule gives changes little: 2.6 and 1.5 planned, 2.5 and 1.6 on the way.
- **Not a rule for other texts or designs.** The floor, the two powers and the best split belong to this text, this design and these sizes.
- **Not a forecast of what a model can do.** The score is surprise at the next letter. It says nothing about any task.

Sources: Kaplan, McCandlish, Henighan, Brown, Chess, Child, Gray, Radford, Wu, Amodei, "Scaling Laws for Neural Language Models", arXiv 2001.08361, 2020. Hoffmann, Borgeaud, Mensch and others, "Training Compute-Optimal Large Language Models", arXiv 2203.15556, 2022. Hestness, Narang, Ardalani and others, "Deep Learning Scaling is Predictable, Empirically", arXiv 1712.00409, 2017. The text: Matt Mahoney, `text8`, http://mattmahoney.net/dc/textdata.html .

## Try it

Step 4 found the rule from the 5 smallest models and forecast the 3 biggest, all with 4 million letters. The biggest miss was 0.017 bits. `exercise.py` repeats the test on the column where every model reads 16 million letters:

```python
floor, power, scale = fit_floor(sizes[:FIT], scores[:FIT])
said = [floor + scale * size ** power for size in sizes[FIT:]]
misses = [s - score for s, score in zip(said, scores[FIT:])]
```

Guess first: is the biggest miss smaller than 0.017 bits, about the same, or several times larger? Then run `python3 exercise.py` (under a second). After that set `FIT = 6` and run it again. Work out why a line through 5 dots can miss more when the models have read more text. A hint: look in the table of step 5 at how much the smallest models still gain from 4 to 16 million letters, and how much the biggest gain.
