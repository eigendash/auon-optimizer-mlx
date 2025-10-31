"""Functional building blocks for AuON: the cosh-RMS brake and Newton-Schulz.

Everything here is a plain function of ``mx.array`` values so that the pieces can
be tested on their own and compared against hand-written references.

The AuON update for a matrix ``G`` (a gradient, or a momentum buffer) is

    x = G / (||G||_F + eps0)
    r = (mean(cosh(gamma * x) ** 2)) ** p
    U = rho * x / (r + eps)

with ``gamma = N ** temperature_alpha`` (``N = numel(G)``), ``p`` the brake
exponent and ``rho`` the spectral trust-region radius.  ``r >= 1`` always, and
``||x||_2 <= ||x||_F <= 1``, so ``||U||_2 < rho`` follows directly; see
:func:`cosh_rms_update`.

The paper writes the same object in two normalisations: vanilla AuON uses
``gamma = 1`` and ``r = sqrt(mean(cosh(x)**2))`` (section 3.1), while the
temperature-scaled variant used for the language-model runs uses
``gamma = N ** 0.48`` and raises the statistic to a power (section 3.3, section
4.2).  Both are the same three lines of code here.
"""

from __future__ import annotations

import mlx.core as mx

#: Numerical guards as written in the paper (sections 3.1, 3.3, 3.4).
EPS0 = 1e-7  # guard on the Frobenius normalisation
EPS = 1e-8  # guard on the brake denominator

#: Newton-Schulz coefficients, as lists of ``(a, b, c)`` triples cycled over the
#: iterations.  ``NS_SIMPLE`` is the single triple used by Muon (Keller Jordan's
#: ``zeropower_via_newtonschulz5``, MLX's built-in ``optimizers.Muon`` and the
#: reference implementation linked from the AuON paper).  ``NS_QUINTIC`` is the
#: per-iteration set used in modded-nanogpt and ``NS_CLASSIC`` is the textbook
#: quintic that actually converges to a polar factor.  The difference matters
#: and is measured in the tests: see :func:`newton_schulz`.
NS_SIMPLE = ((3.4445, -4.7750, 2.0315),)
NS_QUINTIC = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)
NS_CLASSIC = ((1.8750, -1.2500, 0.3750),)
#: The set Muon and Hybrid-AuON use here.
NS_COEFFS = NS_SIMPLE


def normalized_momentum(m: mx.array, eps0: float = EPS0) -> mx.array:
    """``x = m / (||m||_F + eps0)`` -- the unit-Frobenius direction of ``m``."""
    return m / (mx.linalg.norm(m) + eps0)


def cosh_rms(m: mx.array, gamma: float = 1.0, eps: float = EPS) -> mx.array:
    """The brake statistic ``sqrt(mean(cosh(gamma * x) ** 2))`` of x = m/||m||.

    ``gamma`` is the temperature (``N ** alpha`` in the paper).  With
    ``gamma = 0`` every argument is zero, ``cosh`` is 1 and the statistic is
    exactly 1, which is the neutral element of the scaling.
    """
    x = normalized_momentum(m)
    return mx.sqrt(mx.mean(mx.cosh(gamma * x) ** 2) + eps)


def cosh_rms_update(
    m: mx.array,
    *,
    rho: float = 1.0,
    temperature_alpha: float = 0.0,
    brake_exponent: float = 0.5,
    gamma: float | None = None,
    eps0: float = EPS0,
    eps: float = EPS,
) -> tuple[mx.array, mx.array]:
    """The AuON rescaling of ``m``.  Returns ``(U, r)``.

    ``gamma`` overrides the temperature; otherwise ``gamma = numel(m) **
    temperature_alpha``.  The sequence of operations is fixed: normalise,
    evaluate ``cosh`` of the temperature-scaled normalised update, average,
    raise to ``brake_exponent``, divide.  There is no iteration count.

    Spectral trust region: ``mean(cosh(z) ** 2) >= 1`` for every ``z``, so
    ``r >= 1`` and ``||U||_2 = rho * ||x||_2 / (r + eps) <= rho * ||x||_F / (1 +
    eps) < rho``.  The bound is tight: for a rank-one matrix whose entries all
    have magnitude ``1 / sqrt(N)`` the normalised update satisfies
    ``||x||_2 = ||x||_F = 1`` and ``r -> 1`` as ``N`` grows.

    Identity setting: ``gamma = 0`` (all ``cosh`` arguments zero) or
    ``brake_exponent = 0`` (``r = 1`` exactly) reduce the update to plain
    normalised momentum ``rho * m / (||m||_F + eps0) / (1 + eps)``.
    """
    x = m / (mx.linalg.norm(m) + eps0)
    if gamma is None:
        gamma = float(m.size) ** temperature_alpha
    s = mx.mean(mx.cosh(gamma * x) ** 2)
    r = s**brake_exponent
    return rho * x / (r + eps), r


def newton_schulz(
    g: mx.array,
    steps: int = 5,
    eps0: float = EPS0,
    coeffs: tuple[tuple[float, float, float], ...] = NS_COEFFS,
) -> mx.array:
    """Approximate polar factor of a 2-D matrix by Newton-Schulz iteration.

    The iteration runs on the transposed ("tall") orientation and is seeded with
    the Frobenius-normalised matrix, exactly as in Muon.  Each step applies the
    polynomial ``sigma -> a * sigma + b * sigma**3 + c * sigma**5`` to the
    singular values, with ``(a, b, c)`` cycled through ``coeffs``.

    What comes out depends on the coefficients.  With the single Muon triple
    (``NS_SIMPLE``) five steps do **not** drive the singular values to 1: on
    random matrices they land spread around 1 (measured 0.68 to 1.13 for 16x8
    and 24x24 inputs, giving ``||Q.T @ Q - I||_F`` of 0.36 to 0.45).  That is a
    property of the standard Muon coefficients, not a bug here: the NVIDIA
    emerging-optimizers documentation states the iteration "does not produce
    ``U V^T`` but rather something like ``U S' V^T`` where ``S'`` is diagonal
    with noisy values around 1".  With the per-iteration ``NS_QUINTIC`` set the
    residual drops to a few percent, and with the textbook ``NS_CLASSIC``
    quintic repeated it converges to machine precision (for inputs whose
    smallest singular values are not tiny).  The repo's Muon baseline uses
    ``NS_SIMPLE``, i.e. the same operator as reference Muon implementations.
    """
    if g.ndim != 2:
        raise ValueError(f"newton_schulz expects a 2-D matrix, got shape {g.shape}")
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")
    transposed = g.shape[-2] > g.shape[-1]
    x = g.T if transposed else g
    x = x / (mx.linalg.norm(x) + eps0)
    for index in range(steps):
        a, b, c = coeffs[index % len(coeffs)]
        aa = x @ x.T
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x
    return x.T if transposed else x


def muon_scale(g: mx.array) -> float:
    """Muon's shape-dependent learning-rate factor ``sqrt(max(1, m/n))``."""
    return max(1.0, g.shape[-2] / g.shape[-1]) ** 0.5


def rms(a: mx.array) -> mx.array:
    """Root-mean-square of the entries of ``a``."""
    return mx.sqrt(mx.mean(a**2))


def spectral_norm(a: mx.array, iters: int = 100, seed: int = 0) -> mx.array:
    """Largest singular value of ``a`` by power iteration on ``a.T @ a``.

    Used by the tests to check the trust-region bound without calling SVD on
    every random matrix; ``iters = 100`` is far past convergence for the sizes
    used here.  The starting vector comes from an explicit ``mx.random.key`` so
    that measuring a norm never perturbs the global RNG stream.
    """
    n = a.shape[-1]
    v = mx.random.normal(shape=(n,), key=mx.random.key(seed))
    v = v / (mx.linalg.norm(v) + 1e-30)
    for _ in range(iters):
        u = a @ v
        u = u / (mx.linalg.norm(u) + 1e-30)
        v = a.T @ u
        v = v / (mx.linalg.norm(v) + 1e-30)
    return mx.linalg.norm(a @ v)
