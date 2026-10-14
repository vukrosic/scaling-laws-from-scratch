"""Train one model on one amount of text and report its test loss."""
import math
import time

import numpy as np
import torch

from model import LETTERS, WINDOW, Model, loss_of

BATCH = 64                 # windows per step
STEP = BATCH * WINDOW      # letters read per step: 4,096
CHECK_FROM = 90_000_000    # letters used to pick the step size
TEST_FROM = 95_000_000     # the last 5 million letters are the test
TEST_BATCHES = 128         # about half a million test letters

_text = None


def text():
    """All of text8 as numbers: space is 0, a is 1, z is 26."""
    global _text
    if _text is None:
        raw = np.fromfile("text8", dtype=np.uint8)
        assert raw.size == 100_000_000, "text8 not found, see README"
        _text = torch.from_numpy(np.where(raw == 32, 0, raw - 96))
        assert int(_text.max()) == LETTERS - 1
    return _text


def windows(first, count):
    """Cut `count` windows of 65 letters, starting at letter `first`.

    Each window shares its last letter with the next one, so every
    letter is guessed exactly once.
    """
    starts = first + WINDOW * torch.arange(count)
    reach = torch.arange(WINDOW + 1)
    return text()[starts[:, None] + reach[None, :]].long()


def peak_rate(width):
    """How big a nudge each step makes, at the most.

    Wide models need smaller nudges. sweep.py also tries half and
    double of this and keeps the best, see `tries` there.
    """
    return min(0.015, 0.8 / width)


def rate_at(step, steps, peak):
    """Climb for the first tenth, then glide down to a tenth of peak."""
    warm = max(1, steps // 10)
    if step < warm:
        return peak * (step + 1) / warm
    done = (step - warm) / max(1, steps - warm)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * done)))


@torch.no_grad()
def test_loss(model, device, batches=TEST_BATCHES, first=TEST_FROM):
    data = windows(first, batches * BATCH).to(device)
    total = 0.0
    for i in range(batches):
        total += loss_of(model, data[i * BATCH:(i + 1) * BATCH]).item()
    return total / batches


def run(width, layers, letters, seed=0, device="cpu", peak=None,
        keep=False):
    """Read the first `letters` letters of text8 once, then test."""
    torch.manual_seed(seed)
    model = Model(width, layers).to(device)
    peak = peak or peak_rate(width)
    tuner = torch.optim.Adam(
        model.parameters(), lr=peak, betas=(0.9, 0.95))
    steps = letters // STEP
    data = windows(0, steps * BATCH)
    order = torch.randperm(steps * BATCH)
    marks = {steps}
    mark = 61
    while mark < steps:
        marks.add(mark)
        mark *= 2
    curve, recent, began = [], [], time.time()
    for step in range(steps):
        for group in tuner.param_groups:
            group["lr"] = rate_at(step, steps, peak)
        pick = order[step * BATCH:(step + 1) * BATCH]
        loss = loss_of(model, data[pick].to(device))
        tuner.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        tuner.step()
        recent.append(loss.item())
        if step + 1 in marks:
            last = step + 1 == steps
            curve.append({
                "letters": (step + 1) * STEP,
                "train": sum(recent[-50:]) / len(recent[-50:]),
                "test": test_loss(
                    model, device, TEST_BATCHES if last else 32),
            })
    result = {
        "width": width, "layers": layers, "size": model.size(),
        "letters": steps * STEP, "steps": steps, "seed": seed,
        "peak": peak, "loss": curve[-1]["test"],
        "check": test_loss(model, device, 32, CHECK_FROM),
        "curve": curve,
        "seconds": round(time.time() - began, 1),
    }
    if keep:
        result["model"] = model
    return result
