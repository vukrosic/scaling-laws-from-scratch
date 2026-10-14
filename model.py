"""A small language model that reads letters and guesses the next one.

Same design as GPT, shrunk: letters go in, a table of 27 chances
comes out for the next letter. Two knobs set its size:
  width  = how many numbers describe each letter inside the model
  layers = how many times those numbers are reworked
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LETTERS = 27   # a to z, and the space
WINDOW = 64    # how many letters the model sees at once


class Layer(nn.Module):
    """One rework: look back at earlier letters, then think."""

    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.ask = nn.Linear(width, 3 * width)
        self.mix = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.wide = nn.Linear(width, 4 * width)
        self.back = nn.Linear(4 * width, width)

    def forward(self, x, mask):
        b, t, w = x.shape
        h = self.heads
        q, k, v = self.ask(self.norm1(x)).split(w, dim=2)
        q = q.view(b, t, h, w // h).transpose(1, 2)
        k = k.view(b, t, h, w // h).transpose(1, 2)
        v = v.view(b, t, h, w // h).transpose(1, 2)
        score = q @ k.transpose(2, 3) / math.sqrt(w // h)
        score = score.masked_fill(mask[:t, :t], float("-inf"))
        look = F.softmax(score, dim=3) @ v
        look = look.transpose(1, 2).reshape(b, t, w)
        x = x + self.mix(look)
        x = x + self.back(F.gelu(self.wide(self.norm2(x))))
        return x


class Model(nn.Module):
    def __init__(self, width, layers):
        super().__init__()
        heads = max(1, width // 16)
        self.letter = nn.Embedding(LETTERS, width)
        self.place = nn.Embedding(WINDOW, width)
        self.layers = nn.ModuleList(
            [Layer(width, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        future = torch.triu(torch.ones(WINDOW, WINDOW), 1).bool()
        self.register_buffer("mask", future)
        self.apply(self.start)

    @staticmethod
    def start(part):
        if isinstance(part, (nn.Linear, nn.Embedding)):
            nn.init.normal_(part.weight, std=0.02)
        if isinstance(part, nn.Linear):
            nn.init.zeros_(part.bias)

    def forward(self, text):
        t = text.shape[1]
        where = torch.arange(t, device=text.device)
        x = self.letter(text) + self.place(where)
        for layer in self.layers:
            x = layer(x, self.mask)
        # the same letter table turns numbers back into chances
        return self.norm(x) @ self.letter.weight.T

    def size(self):
        """Count the numbers, without the two lookup tables.

        The 2020 paper counts the same way: the tables that turn a
        letter or a place into numbers are left out.
        """
        tables = self.letter.weight.numel() + self.place.weight.numel()
        return sum(p.numel() for p in self.parameters()) - tables


def loss_of(model, text):
    """Average surprise at the true next letter. Lower is better."""
    chances = model(text[:, :-1])
    return F.cross_entropy(
        chances.reshape(-1, LETTERS), text[:, 1:].reshape(-1))
