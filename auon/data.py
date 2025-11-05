"""Tasks: a synthetic induction/copy task and a small built-in character corpus.

Both are plain next-token-prediction problems on sequences of token ids, and both
return a loss mask so the reported loss and accuracy only cover the positions
that matter.

The induction task is the classic copy problem: a random sequence is shown,
then a separator, then the same sequence again, and the model is scored only on
the repeated half.  Solving it requires an induction head -- "attend to the
previous occurrence of the current token and copy what followed it" -- so the
attention pattern is load-bearing and the attention-logit telemetry is
meaningful rather than decorative.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

#: A few hundred words written for this repo, so no corpus licence is involved.
CORPUS = (
    "An optimizer decides how a model moves through its loss landscape. "
    "Vector methods such as Adam keep one scale per coordinate, which is cheap "
    "and robust but ignores the matrix structure of a weight. Orthogonal "
    "momentum methods instead treat each weight matrix as a matrix: they "
    "replace the update by its polar factor, so every direction moves with the "
    "same spectral gain. That equalises the step, but it also throws away the "
    "singular values, and computing a polar factor by SVD or QR is expensive. "
    "Muon keeps the idea and drops the cost: apply momentum first, then "
    "approximate the polar factor with a few Newton-Schulz iterations, which "
    "are just matrix products. The result is fast and it trains well, but a "
    "semi-orthogonal update has unit spectral norm whatever the gradient looks "
    "like, so a layer with a spiky, heavy-tailed gradient is still pushed with "
    "full force. AuON asks what would happen if the update kept its internal "
    "proportions and only its overall size were set by how spiky it is. A "
    "hyperbolic cosine measures spikiness: it is nearly quadratic near zero and "
    "grows exponentially in the tail, so one outlier can shrink the whole step. "
    "That is the emergency brake. The step stays inside a spectral trust region, "
    "the internal ratios of the update are untouched, and the cost is a handful "
    "of elementwise passes over the matrix instead of a polynomial number of "
    "matrix products. Whether that is enough to match orthogonal momentum at "
    "scale is an empirical question, and this repo only answers it at toy scale."
)


@dataclass
class Batch:
    tokens: mx.array  # (B, T) int32
    targets: mx.array  # (B, T) int32
    mask: mx.array  # (B, T) float32


def _to_batch(tokens: np.ndarray, mask: np.ndarray) -> Batch:
    targets = np.zeros_like(tokens)
    targets[:, :-1] = tokens[:, 1:]
    return Batch(
        tokens=mx.array(tokens.astype(np.int32)),
        targets=mx.array(targets.astype(np.int32)),
        mask=mx.array(mask.astype(np.float32)),
    )


class InductionTask:
    """[BOS] a [SEP] a -- scored only on the second copy of ``a``.

    Vocabulary: ``0 .. vocab_size - 1`` are content tokens, ``vocab_size`` is
    BOS and ``vocab_size + 1`` is SEP.
    """

    def __init__(self, vocab_size: int = 32, repeat_len: int = 16):
        self.vocab_size = vocab_size
        self.repeat_len = repeat_len
        self.bos = vocab_size
        self.sep = vocab_size + 1
        self.n_tokens = vocab_size + 2
        self.seq_len = 2 * repeat_len + 2

    def batch(self, rng: np.random.Generator, batch_size: int) -> Batch:
        length = self.repeat_len
        content = rng.integers(0, self.vocab_size, size=(batch_size, length))
        tokens = np.empty((batch_size, self.seq_len), dtype=np.int64)
        tokens[:, 0] = self.bos
        tokens[:, 1 : 1 + length] = content
        tokens[:, 1 + length] = self.sep
        tokens[:, 2 + length :] = content
        mask = np.zeros((batch_size, self.seq_len), dtype=np.float32)
        # Position 1 + length predicts content[:, 0]; position 2 * length
        # predicts content[:, -1].  The final position has no target.
        mask[:, 1 + length : 2 * length + 1] = 1.0
        return _to_batch(tokens, mask)

    def fixed_eval(self, batch_size: int = 256, seed: int = 1234) -> Batch:
        return self.batch(np.random.default_rng(seed), batch_size)


class TinyCorpus:
    """Sliding windows over :data:`CORPUS`, one character per token."""

    def __init__(self, seq_len: int = 64, batch_size: int = 16):
        chars = sorted(set(CORPUS))
        self.vocab = {c: i for i, c in enumerate(chars)}
        self.n_tokens = len(chars)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.data = np.array([self.vocab[c] for c in CORPUS], dtype=np.int64)

    def batch(self, rng: np.random.Generator, batch_size: int | None = None) -> Batch:
        batch_size = batch_size or self.batch_size
        starts = rng.integers(0, len(self.data) - self.seq_len - 1, size=batch_size)
        tokens = np.stack([self.data[s : s + self.seq_len + 1] for s in starts])
        targets = tokens[:, 1:]
        tokens = tokens[:, :-1]
        mask = np.ones_like(tokens, dtype=np.float32)
        return Batch(
            tokens=mx.array(tokens.astype(np.int32)),
            targets=mx.array(targets.astype(np.int32)),
            mask=mx.array(mask.astype(np.float32)),
        )

    def fixed_eval(self, batch_size: int = 8, seed: int = 4321) -> Batch:
        return self.batch(np.random.default_rng(seed), batch_size)


def next_token_accuracy(logits: mx.array, targets: mx.array, mask: mx.array) -> float:
    """Fraction of masked positions whose argmax is the target token."""
    predicted = mx.argmax(logits, axis=-1)
    correct = (predicted == targets).astype(mx.float32) * mask
    value = mx.sum(correct) / mx.maximum(mx.sum(mask), 1.0)
    mx.eval(value)
    return float(value)
