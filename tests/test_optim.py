"""Optimizer-level invariants: the pipeline, the trust region, the baselines.

The two that carry the most weight are that a single AuON step is a fixed
sequence of operations that reduces *exactly* to normalised momentum when the
nonlinear scaling is neutralised, and that the spectral norm of every applied
update stays inside the configured trust region on random matrices.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx.utils import tree_flatten

from auon.normalizers import EPS, EPS0, spectral_norm
from auon.optim import (
    TEMPERATURE_SCALED,
    VANILLA,
    AuON,
    HybridAuON,
    Muon,
    build_optimizer,
    is_hidden_matrix,
    split_optimizer,
)

mx.random.seed(0)


def matrix(rows: int, cols: int, seed: int = 0, scale: float = 1.0) -> mx.array:
    return mx.random.normal(shape=(rows, cols), key=mx.random.key(seed)) * scale


def test_auon_has_no_iteration_count_and_only_a_momentum_buffer():
    """AuON is a fixed sequence of ops: no inner loop, state is just momentum."""
    optimizer = AuON(learning_rate=0.1)
    assert not hasattr(optimizer, "ns_steps")
    parameter = matrix(4, 3, seed=1)
    updated = optimizer.apply_gradients({"w": parameter}, {"w": parameter})
    mx.eval(updated["w"])
    assert set(optimizer.state.keys()) == {"step", "learning_rate", "w"}
    assert set(optimizer.state["w"].keys()) == {"m"}
    assert optimizer.state["w"]["m"].shape == parameter.shape
    # Hybrid-AuON is the variant that does have an iteration count
    assert hasattr(HybridAuON(learning_rate=0.1), "ns_steps")


def test_single_step_matches_a_numpy_reimplementation_of_the_paper_equations():
    """The exact pipeline of sections 3.1/3.3, written out twice."""
    g = 0.05 * np.array([[1.0, -2.0, 3.5], [4.0, 0.5, -1.5]], dtype=np.float32)
    w0 = np.zeros_like(g)
    lr, momentum, rho, wd = 0.3, 0.9, 0.7, 0.01
    alpha, exponent = 0.48, 1.75

    optimizer = AuON(
        learning_rate=lr,
        momentum=momentum,
        weight_decay=wd,
        rho=rho,
        temperature_alpha=alpha,
        brake_exponent=exponent,
    )
    first = optimizer.apply_gradients({"w": mx.array(g)}, {"w": mx.array(w0)})["w"]
    mx.eval(first)

    # step 1: momentum, 2: decoupled decay, 3-7: the AuON rescaling
    m = momentum * 0.0 + (1 - momentum) * g
    x = m / (np.linalg.norm(m) + EPS0)
    gamma = float(m.size) ** alpha
    r = float(np.mean(np.cosh(gamma * x) ** 2)) ** exponent
    update = rho * x / (r + EPS)
    expected = w0 * (1 - lr * wd) - lr * update
    assert np.allclose(np.array(first), expected, atol=1e-6)

    # a second step has to accumulate momentum identically
    g2 = 0.05 * np.array([[0.3, 2.0, -0.7], [1.5, -3.0, 0.25]], dtype=np.float32)
    second = optimizer.apply_gradients({"w": mx.array(g2)}, {"w": first})["w"]
    mx.eval(second)
    m2 = momentum * m + (1 - momentum) * g2
    x2 = m2 / (np.linalg.norm(m2) + EPS0)
    r2 = float(np.mean(np.cosh(gamma * x2) ** 2)) ** exponent
    expected2 = np.array(first) * (1 - lr * wd) - lr * (rho * x2 / (r2 + EPS))
    assert np.allclose(np.array(second), expected2, atol=1e-6)


@pytest.mark.parametrize("neutral", ["gamma", "exponent", "both"])
def test_neutralised_scaling_reduces_to_plain_normalised_momentum(neutral):
    """The documented identity setting, at the level of the whole optimizer."""
    momentum, lr, steps = 0.9, 0.05, 6
    optimizer = AuON(
        learning_rate=lr,
        momentum=momentum,
        rho=1.0,
        gamma=0.0 if neutral in ("gamma", "both") else 1.0,
        brake_exponent=0.0 if neutral in ("exponent", "both") else 0.5,
    )
    weight = np.array(matrix(5, 4, seed=2))
    reference = weight.copy()
    live = mx.array(weight)
    buffer = np.zeros_like(weight)
    for step in range(steps):
        g = 0.1 * np.array(matrix(5, 4, seed=20 + step))
        live = optimizer.apply_gradients({"w": mx.array(g)}, {"w": live})["w"]
        mx.eval(live)
        buffer = momentum * buffer + (1 - momentum) * g
        reference = reference - lr * buffer / (np.linalg.norm(buffer) + EPS0) / (1.0 + EPS)
    assert np.allclose(np.array(live), reference, atol=1e-6)


@pytest.mark.parametrize("rho", [1.0, 0.3])
@pytest.mark.parametrize("arm", ["auon", "auon_temp", "hybrid_auon"])
def test_applied_update_stays_inside_the_spectral_trust_region(arm, rho):
    """||W_new - W_old||_2 <= rho at every step, by power iteration."""
    shapes = [(6, 6), (16, 8), (8, 16), (32, 32)]
    for rows, cols in shapes:
        optimizer = (
            AuON(learning_rate=0.05, rho=rho, **TEMPERATURE_SCALED)
            if arm == "auon_temp"
            else HybridAuON(learning_rate=0.05, rho=rho, **TEMPERATURE_SCALED)
            if arm == "hybrid_auon"
            else AuON(learning_rate=0.05, rho=rho, **VANILLA)
        )
        weight = mx.zeros((rows, cols))
        for step in range(4):
            g = matrix(rows, cols, seed=100 * rows + step, scale=1.0 + step)
            updated = optimizer.apply_gradients({"w": g}, {"w": weight})
            mx.eval(updated["w"])
            delta = updated["w"] - weight
            norm = float(spectral_norm(delta, iters=200, seed=step))
            assert norm <= rho * (1 + 1e-4), (arm, rows, cols, step, norm)
            weight = updated["w"]


def test_trust_region_bound_is_tight_for_a_large_dense_update():
    """rho is a real bound, not a loose one: a rank-one gradient reaches it."""
    rows = cols = 100
    optimizer = AuON(
        learning_rate=1.0, momentum=0.0, rho=1.0, gamma=1.0, brake_exponent=0.5
    )
    weight = mx.zeros((rows, cols))
    gradient = mx.ones((rows, cols)) / rows
    updated = optimizer.apply_gradients({"w": gradient}, {"w": weight})
    mx.eval(updated["w"])
    delta = weight - updated["w"]  # the update, sign removed
    norm = float(spectral_norm(delta, iters=400))
    assert norm <= 1.0
    assert norm > 0.999


def test_rho_scales_every_applied_update_linearly():
    g = matrix(9, 5, seed=4)
    d = {}
    for rho in (1.0, 0.5, 0.25):
        optimizer = AuON(learning_rate=1.0, momentum=0.0, rho=rho)
        w = mx.zeros((9, 5))
        d[rho] = np.array(w - optimizer.apply_gradients({"w": g}, {"w": w})["w"])
    assert np.allclose(d[0.5], 0.5 * d[1.0], atol=1e-7)
    assert np.allclose(d[0.25], 0.25 * d[1.0], atol=1e-7)


def test_hybrid_auon_orthogonalises_matrices_only():
    g2d = matrix(16, 8, seed=5)
    g1d = matrix(1, 16, seed=6).reshape((16,))
    plain = lambda: AuON(learning_rate=1.0, momentum=0.0, **VANILLA)
    hybrid = lambda: HybridAuON(learning_rate=1.0, momentum=0.0, ns_steps=5, **VANILLA)
    a2 = plain().apply_gradients({"w": g2d}, {"w": mx.zeros((16, 8))})["w"]
    h2 = hybrid().apply_gradients({"w": g2d}, {"w": mx.zeros((16, 8))})["w"]
    a1 = plain().apply_gradients({"b": g1d}, {"b": mx.zeros((16,))})["b"]
    h1 = hybrid().apply_gradients({"b": g1d}, {"b": mx.zeros((16,))})["b"]
    mx.eval(a2, h2, a1, h1)
    assert float(mx.max(mx.abs(a1 - h1))) < 1e-7  # 1-D untouched by Newton-Schulz
    assert float(mx.max(mx.abs(a2 - h2))) > 1e-3  # 2-D transformed


def test_muon_matches_the_mlx_builtin_muon_step_for_step():
    """Two independent implementations of the same baseline must agree."""
    from mlx.optimizers import Muon as MLXMuon

    for shape in ((8, 4), (4, 8), (6, 6)):
        mine = Muon(learning_rate=0.02, momentum=0.9, weight_decay=0.0, ns_steps=5)
        theirs = MLXMuon(learning_rate=0.02, momentum=0.9, weight_decay=0.0, ns_steps=5)
        w_mine = mx.zeros(shape)
        w_theirs = mx.zeros(shape)
        for step in range(3):
            g = matrix(*shape, seed=50 + step)
            w_mine = mine.apply_gradients({"w": g}, {"w": w_mine})["w"]
            w_theirs = theirs.apply_gradients({"w": g}, {"w": w_theirs})["w"]
            mx.eval(w_mine, w_theirs)
        assert float(mx.max(mx.abs(w_mine - w_theirs))) < 1e-5, shape


def test_muon_update_is_approximately_semi_orthogonal():
    """The baseline's update has near-unit spectral norm whatever the gradient."""
    weight = mx.zeros((16, 16))
    base = matrix(16, 16, seed=60)
    norms = []
    for scale in (1e-3, 1.0, 1e3):
        optimizer = Muon(learning_rate=0.01, momentum=0.0, weight_decay=0.0, ns_steps=5)
        updated = optimizer.apply_gradients({"w": base * scale}, {"w": weight})
        mx.eval(updated["w"])
        delta = updated["w"] - weight
        norms.append(float(spectral_norm(delta, iters=300)) / 0.01)
    # approximately unit spectral norm, and identical for gradients that differ
    # by six orders of magnitude: Muon ignores the gradient distribution
    assert all(0.9 <= n <= 1.3 for n in norms)
    assert max(norms) - min(norms) < 1e-4


def test_muon_leaves_one_dimensional_parameters_alone():
    optimizer = Muon(learning_rate=0.1, momentum=0.0, weight_decay=0.0)
    g = matrix(1, 6, seed=7).reshape((6,))
    w = mx.zeros((6,))
    updated = optimizer.apply_gradients({"b": g}, {"b": w})["b"]
    mx.eval(updated)
    assert np.allclose(np.array(updated), -0.1 * np.array(g), atol=1e-6)


def test_weight_decay_is_decoupled_and_does_not_touch_the_momentum():
    optimizer = AuON(learning_rate=0.5, momentum=0.0, weight_decay=0.1)
    w = mx.full((4, 4), 2.0)
    updated = optimizer.apply_gradients({"w": mx.zeros((4, 4))}, {"w": w})["w"]
    mx.eval(updated)
    assert np.allclose(np.array(updated), 2.0 * (1 - 0.5 * 0.1), atol=1e-6)


def test_huge_gradient_is_damped_and_never_produces_nan():
    """The emergency brake in one line: spiky in, tiny out, always finite."""
    optimizer = AuON(learning_rate=1.0, momentum=0.0, **TEMPERATURE_SCALED)
    weight = mx.zeros((64, 64))
    update = mx.zeros((64, 64))
    for spike in (0.0, 10.0, 1e6):
        g = matrix(64, 64, seed=8)
        g = g.at[0, 0].add(spike)
        updated = optimizer.apply_gradients({"w": g}, {"w": weight})["w"]
        mx.eval(updated)
        update = weight - updated["w"] if False else updated
        assert bool(mx.all(mx.isfinite(updated)))
    # with an extreme spike the update collapses; Muon's does not
    g = matrix(64, 64, seed=8).at[0, 0].add(1e6)
    auon_delta = AuON(learning_rate=1.0, momentum=0.0, **TEMPERATURE_SCALED).apply_gradients(
        {"w": g}, {"w": weight}
    )["w"]
    muon_delta = Muon(learning_rate=1.0, momentum=0.0, weight_decay=0.0).apply_gradients(
        {"w": g}, {"w": weight}
    )["w"]
    mx.eval(auon_delta, muon_delta)
    assert float(spectral_norm(auon_delta, iters=200)) < 1e-3
    assert float(spectral_norm(muon_delta, iters=200)) > 0.5


def test_invalid_hyperparameters_are_rejected():
    with pytest.raises(ValueError):
        AuON(learning_rate=0.1, rho=0.0)
    with pytest.raises(ValueError):
        AuON(learning_rate=0.1, momentum=1.5)
    with pytest.raises(ValueError):
        AuON(learning_rate=0.1, weight_decay=-1.0)
    with pytest.raises(ValueError):
        HybridAuON(learning_rate=0.1, ns_steps=0)
    with pytest.raises(ValueError):
        build_optimizer("not-an-optimizer", 0.1)


def test_parameter_split_matches_the_papers_adam_group():
    assert is_hidden_matrix("blocks.0.attn.qkv.weight", mx.zeros((8, 4)))
    assert is_hidden_matrix("blocks.1.fc2.weight", mx.zeros((8, 4)))
    assert not is_hidden_matrix("tok_emb.weight", mx.zeros((8, 4)))
    assert not is_hidden_matrix("pos_emb.weight", mx.zeros((8, 4)))
    assert not is_hidden_matrix("head.weight", mx.zeros((8, 4)))
    assert not is_hidden_matrix("blocks.0.norm1.weight", mx.zeros((8,)))


def test_split_optimizer_routes_each_parameter_to_the_right_rule():
    # MLX's MultiOptimizer flattens the parameter tree, so both halves of the
    # split need at least one entry
    parameters = {"hidden": {"weight": matrix(4, 4, seed=1)}, "head": {"weight": matrix(4, 4, seed=2)}}
    gradients = {
        "hidden": {"weight": matrix(4, 4, seed=3)},
        "head": {"weight": matrix(4, 4, seed=4)},
    }
    frozen_matrix_rule = AuON(learning_rate=0.0, momentum=0.0)
    adam = build_optimizer("adam", 0.0, adam_learning_rate=0.1)
    updated = split_optimizer(frozen_matrix_rule, adam).apply_gradients(gradients, parameters)
    mx.eval(updated["hidden"]["weight"], updated["head"]["weight"])
    assert np.allclose(np.array(updated["hidden"]["weight"]), np.array(parameters["hidden"]["weight"]))
    assert not np.allclose(np.array(updated["head"]["weight"]), np.array(parameters["head"]["weight"]))


@pytest.mark.parametrize(
    "arm", ["adam", "sgd_momentum", "muon", "auon", "auon_temp", "hybrid_auon"]
)
def test_every_arm_stays_finite_and_bounded_under_extreme_gradients(arm):
    """1e9-sized gradients: everything stays finite, and the AuON arms keep
    ||dW||_2 <= lr * rho, which is the trust region doing its job."""
    learning_rate = 0.05
    optimizer = build_optimizer(arm, learning_rate)

    # a list-valued path so that MLX's flatten/unflatten round-trip agrees with
    # the structure the split hands back
    def make(scale):
        return {
            "blocks": [{"attn": {"qkv": {"weight": matrix(8, 8, seed=11, scale=scale)}}}],
            "head": {"weight": matrix(8, 8, seed=12, scale=scale)},
        }

    weights = make(0.0)
    for scale in (1.0, 1e6, 1e9):
        previous = dict(tree_flatten(weights))
        weights = optimizer.apply_gradients(make(scale), weights)
        for name, value in tree_flatten(weights):
            mx.eval(value)
            assert bool(mx.all(mx.isfinite(value))), (arm, scale, name)
            delta = float(spectral_norm(value - previous[name], iters=200))
            if arm in ("auon", "auon_temp", "hybrid_auon"):
                assert delta <= learning_rate * 1.001, (arm, scale, name, delta)
            elif arm == "muon":
                assert delta <= 0.5, (arm, scale, name, delta)


def test_muon_shape_scaling_matches_sqrt_of_the_aspect_ratio():
    """A 16x4 update moves sqrt(4) = 2x further than a 4x16 one, all else equal."""
    upright = lambda: Muon(learning_rate=1.0, momentum=0.0, weight_decay=0.0, nesterov=False)
    tall = mx.zeros((16, 4))
    wide = mx.zeros((4, 16))
    g = matrix(16, 4, seed=12)
    tall_update = upright().apply_gradients({"w": g}, {"w": tall})["w"]
    wide_update = upright().apply_gradients({"w": g.T}, {"w": wide})["w"]
    mx.eval(tall_update, wide_update)
    ratio = float(mx.linalg.norm(tall_update)) / float(mx.linalg.norm(wide_update))
    assert ratio == pytest.approx(2.0, rel=1e-4)
