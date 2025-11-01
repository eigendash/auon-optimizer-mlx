"""A tiny pre-norm transformer language model with attention-logit telemetry.

Two blocks, learned positional embeddings, RMSNorm and a plain GELU MLP.  The
only unusual thing is that :meth:`TinyLM.__call__` can be asked to append the
largest absolute *pre-softmax* attention logit it produced in each block, which
is the quantity the paper tracks when it talks about exploding attention logits
(section 3.5).  Masked (future) entries are excluded, otherwise the ``-inf``
mask would dominate the statistic.
"""

from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

NEG_INF = -1e9


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} is not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def __call__(self, x: mx.array, mask: mx.array, keep: mx.array, capture=None):
        b, t, _ = x.shape
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        shape = (b, t, self.n_heads, self.head_dim)
        # (B, H, T, Dh)
        q = q.reshape(shape).transpose(0, 2, 1, 3)
        k = k.reshape(shape).transpose(0, 2, 1, 3)
        v = v.reshape(shape).transpose(0, 2, 1, 3)
        logits = (q @ k.transpose(0, 1, 3, 2)) * self.scale + mask
        if capture is not None:
            capture.append(mx.max(mx.abs(logits) * keep))
        weights = mx.softmax(logits, axis=-1)
        y = (weights @ v).transpose(0, 2, 1, 3).reshape(b, t, -1)
        return self.out(y)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads)
        self.norm2 = nn.RMSNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def __call__(self, x, mask, keep, capture=None):
        x = x + self.attn(self.norm1(x), mask, keep, capture)
        h = self.fc1(self.norm2(x))
        return x + self.fc2(nn.gelu(h))


class TinyLM(nn.Module):
    """Character/token-level decoder-only LM.

    Args:
        vocab_size: vocabulary size.
        d_model, n_heads, n_layers, d_ff: the usual widths.
        max_seq_len: length of the learned positional table.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 96,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = [Block(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def num_params(self) -> int:
        return sum(w.size for _, w in tree_flatten(self.trainable_parameters()))

    def __call__(self, tokens: mx.array, capture: list | None = None):
        b, t = tokens.shape
        if t > self.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds max_seq_len {self.max_seq_len}")
        positions = mx.arange(t)
        x = self.tok_emb(tokens) + self.pos_emb(positions)[None]
        keep = mx.tril(mx.ones((t, t)))[None, None]
        mask = (1.0 - keep) * NEG_INF
        for block in self.blocks:
            x = block(x, mask, keep, capture)
        return self.head(self.norm(x))

    def max_attention_logit(self, tokens: mx.array) -> float:
        """Largest absolute causal pre-softmax attention logit in this batch."""
        capture: list = []
        self(tokens, capture=capture)
        value = mx.max(mx.stack(capture))
        mx.eval(value)
        return float(value)


def cross_entropy(logits: mx.array, targets: mx.array, mask: mx.array) -> mx.array:
    """Token-mean cross entropy over the positions selected by ``mask``."""
    log_probs = nn.log_softmax(logits, axis=-1)
    picked = mx.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
    mask = mask.astype(picked.dtype)
    total = mx.sum(mask)
    picked = mx.where(mask > 0, picked, 0.0)
    return -mx.sum(picked * mask) / mx.maximum(total, 1.0)
