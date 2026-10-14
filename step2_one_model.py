"""Step 2. The smallest model: 496 numbers. Train it and test it."""
import math

import torch

from laws import best_runs, load
from model import Model
from sweep import write
from train import STEP, test_loss, run

WIDTH = 4
STEPS = 977                 # 977 steps x 4,096 letters = 4 million
device = "mps" if torch.backends.mps.is_available() else "cpu"

torch.manual_seed(0)
fresh = Model(WIDTH, 2).to(device)
print("numbers in the model:", fresh.size())
print("letters it will read:", f"{STEPS * STEP:,}")

before = test_loss(fresh, device) / math.log(2)
print()
print("before training, every number is a random start")
print(f"  score: {before:.2f} bits per letter")
print("  it writes:", repr(write(fresh, device, 60)))

# the step size that won for this run in the sweep
won = [r for r in best_runs(load())
       if (r["width"], r["layers"], r["steps"]) == (WIDTH, 2, STEPS)
       and r["seed"] == 0 and "budget" not in r]
result = run(WIDTH, 2, STEPS * STEP, 0, device, won[0]["peak"],
             keep=True)
after = result["loss"] / math.log(2)
print()
print("after reading 4 million letters once")
print(f"  score: {after:.2f} bits per letter")
print("  it writes:", repr(write(result["model"], device, 60)))
print(f"  ({result['seconds']} seconds)")
