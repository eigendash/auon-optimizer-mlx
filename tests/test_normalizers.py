"""Invariants of the cosh-RMS brake and the Newton-Schulz iteration.

These are the properties the rest of the repository leans on: the spectral
trust-region bound (checked by power iteration, not by trusting the algebra),
its tightness, the exact neutral element of the nonlinear scaling, the tail
damping, and the semi-orthogonality of the Newton-Schulz output.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from auon.normalizers import (
    NS_CLASSIC,
    NS_QUINTIC,
    NS_SIMPLE,
    cosh_rms,
    cosh_rms_update,
    muon_scale,
    newton_schulz,
    normalized_momentum,
    spectral_norm,
)

mx.random.seed(0)


def random_matrix(rows: int, cols: int, scale: float = 1.0, seed: int = 0) -> mx.array:
    key = mx.random.key(seed)
    return mx.random.normal(shape=(rows, cols), key=key) * scale


def gram_residual(q: mx.array) -> float:
    """||Q Q^T - I|| or ||Q^T Q - I||, whichever is the small square one."""
    gram = np.array(q @ q.T) if q.shape[0] <= q.shape[1] else np.array(q.T @ q)
    return float(np.abs(gram - np.eye(gram.shape[0])).max())


def test_normalized_momentum_has_unit_frobenius_norm():
    m = random_matrix(16, 8, scale=5.0)
    x = normalized_momentum(m)
    assert abs(float(mx.linalg.norm(x)) - 1.0) < 1e-5
    # the guards only matter when the matrix is (almost) zero
    tiny = normalized_momentum(mx.zeros((4, 4)))
    assert float(mx.linalg.norm(tiny)) == pytest.approx(0.0)


def test_null_temperature_is_an_exact_identity():
    """``gamma = 0`` makes cosh(0) = 1, so r = 1 and U is normalised momentum."""
    m = random_matrix(8, 5, seed=1)
    update, r = cosh_rms_update(m, rho=1.0, gamma=0.0, brake_exponent=1.75)
    expected = normalized_momentum(m) / (1.0 + 1e-8)
    assert float(r) == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(np.array(update), np.array(expected), atol=1e-6)


def test_zero_brake_exponent_is_an_exact_identity():
    """``r = s ** 0 = 1`` regardless of the temperature."""
    m = random_matrix(8, 5, seed=2)
    update, r = cosh_rms_update(m, rho=1.0, gamma=7.0, brake_exponent=0.0)
    assert float(r) == 1.0
    assert np.allclose(
        np.array(update), np.array(normalized_momentum(m) / (1.0 + 1e-8)), atol=1e-6
    )


def test_rho_scales_the_update_and_is_the_only_source_of_scale():
    m = random_matrix(12, 7, seed=3)
    unit, _ = cosh_rms_update(m, rho=1.0)
    half, _ = cosh_rms_update(m, rho=0.5)
    assert np.allclose(np.array(half), 0.5 * np.array(unit), atol=1e-7)


@pytest.mark.parametrize(
    "config",
    [
        {"gamma": 1.0, "brake_exponent": 0.5},
        {"temperature_alpha": 0.48, "brake_exponent": 1.75},
        {"temperature_alpha": 0.0, "brake_exponent": 3.5},
    ],
)
@pytest.mark.parametrize("rho", [1.0, 0.25])
def test_spectral_norm_of_the_update_never_exceeds_rho(config, rho):
    """||dW||_2 <= rho, verified by power iteration on 24 random shapes."""
    shapes = [(1, 1), (2, 5), (5, 2), (8, 8), (13, 7), (32, 32), (64, 16), (16, 64)]
    for index, shape in enumerate(shapes):
        for seed in range(3):
            m = random_matrix(*shape, scale=10 ** (seed - 1), seed=100 * index + seed)
            update, _ = cosh_rms_update(m, rho=rho, **config)
            norm = float(spectral_norm(update, iters=200, seed=seed))
            assert norm <= rho * (1 + 1e-5), (shape, seed, norm)


def test_spectral_bound_is_tight_for_a_large_dense_matrix():
    """A rank-one matrix with tiny equal entries reaches very close to rho."""
    n = 100  # N = 10_000 entries
    m = mx.ones((n, n)) / n
    rho = 1.0
    update, r = cosh_rms_update(m, rho=rho, gamma=1.0, brake_exponent=0.5)
    norm = float(spectral_norm(update, iters=400, seed=0))
    assert float(r) < 1.001
    assert norm <= rho
    assert norm >= 0.999 * rho  # tight: within 0.1% of the bound
    # the bound is only slack because of the 1e-7 / 1e-8 guards
    assert norm > 1.0 / (1.0 + 1e-8) * 0.999


def test_trust_region_survives_an_adversarial_spike():
    """A huge single entry must not push the update outside the region."""
    m = random_matrix(64, 64, seed=7)
    m = m.at[0, 0].add(1e6)
    for config in (
        {"gamma": 1.0, "brake_exponent": 0.5},
        {"temperature_alpha": 0.48, "brake_exponent": 1.75},
    ):
        update, _ = cosh_rms_update(m, rho=0.5, **config)
        assert bool(mx.all(mx.isfinite(update)))
        assert float(spectral_norm(update, iters=300)) <= 0.5 * (1 + 1e-5)


def test_temperature_scaled_brake_suppresses_a_spike_exponentially():
    """The mechanism the paper calls the emergency brake."""
    base = random_matrix(64, 64, seed=11)
    flat, _ = cosh_rms_update(base, temperature_alpha=0.48, brake_exponent=1.75)
    norms = []
    for spike in (0.0, 1.0, 10.0, 100.0):
        m = base.at[0, 0].add(spike)
        update, r = cosh_rms_update(m, temperature_alpha=0.48, brake_exponent=1.75)
        mx.eval(update, r)
        assert bool(mx.all(mx.isfinite(update)))
        norms.append(float(spectral_norm(update, iters=300)))
    # spikier input, smaller step: monotone, and a collapse at the tail
    assert norms[1] < norms[0]
    assert norms[2] < 1e-5
    assert norms[3] < 1e-5
    assert norms[2] < 1e-4 * float(spectral_norm(flat, iters=300))


def test_vanilla_brake_cannot_shrink_the_step_by_more_than_cosh_one():
    """Why the paper's temperature scaling matters (documented inference).

    With ``gamma = 1`` the argument of cosh is a normalised entry, so
    ``|arg| <= 1`` and the brake factor is bounded by ``cosh(1)**(2 * p)``.
    The exponential suppression of section 3.5 needs ``gamma`` to grow with the
    tensor size; section 3.5.1 uses ``gamma = sqrt(N)`` and sections 3.3/4.2 use
    ``N ** 0.48``.
    """
    limit = math.cosh(1.0)
    for seed in range(10):
        m = random_matrix(32, 32, scale=10.0**seed, seed=seed)
        update, r = cosh_rms_update(m, gamma=1.0, brake_exponent=0.5)
        x = normalized_momentum(m)
        assert float(r) <= limit * (1 + 1e-5)
        assert float(mx.linalg.norm(update)) >= float(mx.linalg.norm(x)) / limit - 1e-6
    # and the statistic really is scale invariant
    m = random_matrix(16, 16, seed=5)
    small, _ = cosh_rms_update(m, gamma=1.0)
    large, _ = cosh_rms_update(m * 1e4, gamma=1.0)
    assert np.allclose(np.array(small), np.array(large), atol=1e-6)


def test_cosh_rms_matches_a_hand_computed_statistic():
    m = mx.array([[3.0, 4.0], [0.0, 0.0]])
    # ||m||_F = 5, x = [0.6, 0.8, 0, 0]
    expected = math.sqrt((math.cosh(0.6) ** 2 + math.cosh(0.8) ** 2 + 1.0 + 1.0) / 4)
    assert float(cosh_rms(m, gamma=1.0, eps=0.0)) == pytest.approx(expected, rel=1e-6)


def test_newton_schulz_with_muon_coefficients_is_approximately_semi_orthogonal():
    """The baseline sanity check: Muon's NS5 output compresses the spectrum.

    The standard Muon coefficient set is documented (NVIDIA emerging-optimizers)
    to give "noisy values around 1" rather than exactly 1, so the assertions are
    about the spectral norm, the energy per direction and the compression of the
    spectrum -- not about a small residual.
    """
    for shape in ((16, 8), (24, 24), (4, 9)):
        g = random_matrix(*shape, seed=3)
        q = newton_schulz(g, steps=5, coeffs=NS_SIMPLE)
        mx.eval(q)
        source = mx.linalg.svd(g, compute_uv=False)
        output = mx.linalg.svd(q, compute_uv=False)
        mx.eval(source, output)
        source = np.array(source)
        output = np.array(output)
        assert 0.9 <= output.max() <= 1.3  # approximately unit spectral norm
        assert output.max() / output.min() < source.max() / source.min()
        assert math.sqrt(float(np.mean(output**2))) == pytest.approx(1.0, abs=0.2)
        assert gram_residual(q) > 0.05  # approximate, not the exact polar factor


def test_optimised_quintic_coefficients_reach_semi_orthogonality_in_five_steps():
    for shape in ((16, 8), (4, 9)):
        q = newton_schulz(random_matrix(*shape, seed=3), steps=5, coeffs=NS_QUINTIC)
        mx.eval(q)
        assert gram_residual(q) < 0.15
        singular = np.linalg.svd(np.array(q), compute_uv=False)
        assert singular.min() > 0.9 and singular.max() < 1.1


def test_textbook_quintic_converges_to_the_polar_factor():
    g = random_matrix(16, 8, seed=3)
    q = newton_schulz(g, steps=20, coeffs=NS_CLASSIC)
    mx.eval(q)
    assert gram_residual(q) < 1e-3
    # and it really is the polar factor of the input
    u, _, vt = np.linalg.svd(np.array(g), full_matrices=False)
    assert np.abs(np.array(q) - u @ vt).max() < 1e-2


def test_newton_schulz_transposes_wide_inputs_correctly():
    for shape in ((4, 9), (9, 4)):
        q = newton_schulz(random_matrix(*shape, seed=4), steps=5, coeffs=NS_CLASSIC)
        mx.eval(q)
        assert q.shape == shape
        assert gram_residual(q) < 1e-3


def test_newton_schulz_matches_the_mlx_builtin_muon_operator():
    """Equivalence with an independent implementation of the same maths."""
    from mlx.optimizers import Muon as MLXMuon

    builtin = MLXMuon(learning_rate=0.1)
    for shape in ((16, 8), (8, 16), (12, 12)):
        g = random_matrix(*shape, seed=9)
        mine = newton_schulz(g, steps=5)
        theirs = builtin._zeropower_via_newtonschulz5(g, steps=5)
        mx.eval(mine, theirs)
        assert float(mx.max(mx.abs(mine - theirs))) < 1e-5


def test_more_newton_schulz_steps_decorrelate_more():
    g = random_matrix(24, 24, seed=6)
    errors = []
    for steps in (1, 3, 20):
        q = newton_schulz(g, steps=steps, coeffs=NS_CLASSIC)
        mx.eval(q)
        errors.append(gram_residual(q))
    assert errors[0] > errors[1] > errors[2]


def test_newton_schulz_rejects_non_matrices():
    with pytest.raises(ValueError):
        newton_schulz(mx.ones((5,)), steps=5)


def test_spectral_norm_agrees_with_svd():
    for shape in ((6, 6), (4, 9), (9, 4)):
        a = random_matrix(*shape, seed=8)
        singular = mx.linalg.svd(a, compute_uv=False)
        mx.eval(singular)
        assert float(spectral_norm(a, iters=500)) == pytest.approx(
            float(singular[0]), rel=1e-3
        )


def test_muon_scale_is_the_documented_shape_factor():
    assert muon_scale(mx.zeros((8, 4))) == pytest.approx(math.sqrt(2.0))
    assert muon_scale(mx.zeros((4, 8))) == pytest.approx(1.0)
    assert muon_scale(mx.zeros((16, 16))) == pytest.approx(1.0)
