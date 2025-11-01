"""The toy transformer and its attention-logit telemetry.

The telemetry is the measurement the paper's stability claim rests on, so it is
checked against a hand-written recomputation of the same attention logits rather
than trusted.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten

from auon.model import TinyLM, cross_entropy


def small_model(**overrides) -> TinyLM:
    config = dict(vocab_size=5, d_model=8, n_heads=2, n_layers=1, d_ff=16, max_seq_len=8)
    config.update(overrides)
    mx.random.seed(0)
    return TinyLM(**config)


def test_parameter_count_is_about_two_hundred_thousand_with_two_blocks():
    mx.random.seed(0)
    model = TinyLM(vocab_size=34, d_model=96, n_heads=4, n_layers=2, d_ff=256, max_seq_len=48)
    assert len(model.blocks) == 2
    assert 150_000 < model.num_params() < 250_000
    assert model.num_params() == sum(w.size for _, w in tree_flatten(model.trainable_parameters()))


def test_attention_logits_are_causal():
    model = small_model()
    first = mx.array([[1, 2, 3, 4]])
    second = mx.array([[1, 2, 3, 0]])
    a = model(first)
    b = model(second)
    mx.eval(a, b)
    # only the last position can see the changed token
    assert np.allclose(np.array(a)[0, :3], np.array(b)[0, :3], atol=1e-6)
    assert not np.allclose(np.array(a)[0, 3], np.array(b)[0, 3], atol=1e-6)


def test_max_attention_logit_matches_a_manual_recomputation():
    model = small_model()
    tokens = mx.array([[1, 2, 3, 4], [0, 4, 1, 2]])
    reported = model.max_attention_logit(tokens)

    block = model.blocks[0]
    x = model.tok_emb(tokens) + model.pos_emb(mx.arange(tokens.shape[1]))[None]
    h = block.norm1(x)
    qkv = h @ block.attn.qkv.weight.T
    q, k, _ = mx.split(qkv, 3, axis=-1)
    shape = (tokens.shape[0], tokens.shape[1], block.attn.n_heads, block.attn.head_dim)
    q = q.reshape(shape).transpose(0, 2, 1, 3)
    k = k.reshape(shape).transpose(0, 2, 1, 3)
    logits = (q @ k.transpose(0, 1, 3, 2)) * block.attn.scale
    length = tokens.shape[1]
    keep = np.tril(np.ones((length, length), dtype=np.float32))
    manual = float(np.abs(np.array(logits) * keep).max())
    assert reported == pytest.approx(manual, rel=1e-5)
    # masked entries never leak into the statistic
    assert reported < 100.0


def test_capture_does_not_change_the_logits():
    model = small_model()
    tokens = mx.array([[1, 2, 3, 4]])
    plain = model(tokens)
    capture: list = []
    with_capture = model(tokens, capture=capture)
    mx.eval(plain, with_capture)
    assert len(capture) == 1
    assert np.allclose(np.array(plain), np.array(with_capture), atol=0.0)


def test_attention_logit_telemetry_tracks_the_weights():
    """Scaling the query projection up scales the reported maximum up."""
    model = small_model()
    tokens = mx.array([[1, 2, 3, 4]])
    before = model.max_attention_logit(tokens)
    block = model.blocks[0]
    block.attn.qkv.weight = block.attn.qkv.weight * 3.0
    after = model.max_attention_logit(tokens)
    assert after > 2.5 * before


def test_sequence_longer_than_the_position_table_raises():
    model = small_model(max_seq_len=4)
    with pytest.raises(ValueError):
        model(mx.array([[1, 2, 3, 4, 0]]))


def test_cross_entropy_only_counts_masked_positions():
    logits = mx.array([[[0.0, 1.0], [1.0, 0.0]]])
    targets = mx.array([[1, 1]])
    mask = mx.array([[1.0, 0.0]])
    loss = cross_entropy(logits, targets, mask)
    mx.eval(loss)
    expected = -np.log(np.exp(1.0) / (np.exp(0.0) + np.exp(1.0)))
    assert float(loss) == pytest.approx(expected, rel=1e-6)
    # weighting both positions equally gives the mean of the two losses
    loss_both = cross_entropy(logits, targets, mx.array([[1.0, 1.0]]))
    mx.eval(loss_both)
    second = -np.log(np.exp(0.0) / (np.exp(0.0) + np.exp(1.0)))
    assert float(loss_both) == pytest.approx((expected + second) / 2, rel=1e-6)


def test_an_all_zero_mask_gives_a_finite_zero_loss():
    logits = mx.zeros((1, 3, 4))
    loss = cross_entropy(logits, mx.zeros((1, 3), dtype=mx.int32), mx.zeros((1, 3)))
    mx.eval(loss)
    assert float(loss) == 0.0


def test_gradients_are_finite_and_non_zero():
    model = small_model()
    tokens = mx.array([[1, 2, 3, 4]])
    targets = mx.array([[2, 3, 4, 0]])
    mask = mx.ones((1, 4))

    def loss_fn(m, t, y, k):
        return cross_entropy(m(t), y, k)

    loss, grads = nn.value_and_grad(model, loss_fn)(model, tokens, targets, mask)
    mx.eval(loss, grads)
    assert np.isfinite(float(loss))
    norms = [float(mx.linalg.norm(g)) for _, g in tree_flatten(grads)]
    assert all(np.isfinite(n) for n in norms)
    assert max(norms) > 0.0
