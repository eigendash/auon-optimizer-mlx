"""AuON, Hybrid-AuON and the Muon baseline, as MLX optimizers.

A single vanilla AuON step on a 2-D parameter ``W`` with gradient ``g`` is the
fixed sequence

1. ``m <- momentum * m + (1 - momentum) * g``      (EMA momentum buffer)
2. ``W <- W * (1 - lr * weight_decay)``            (decoupled weight decay)
3. ``x <- m / (||m||_F + 1e-7)``
4. ``gamma <- numel(W) ** temperature_alpha``      (``1`` when alpha is 0)
5. ``r <- (mean(cosh(gamma * x) ** 2)) ** brake_exponent``
6. ``U <- rho * x / (r + 1e-8)``
7. ``W <- W - lr * U``

There is no inner loop and no iteration-count hyperparameter (tested): the cost
is a handful of elementwise passes plus one reduction, i.e. ``O(numel)``.
Hybrid-AuON inserts ``ns_steps`` Newton-Schulz iterations before step 3, which is
``O(m n min(m, n))`` per iteration; Muon *is* that iteration applied to the
momentum with no cosh brake and no unit-Frobenius renormalisation.
"""

from __future__ import annotations

from typing import Callable, Union

import mlx.core as mx
from mlx import nn
from mlx.optimizers import Adam, MultiOptimizer, Optimizer

from .normalizers import (
    EPS,
    EPS0,
    NS_SIMPLE,
    cosh_rms_update,
    muon_scale,
    newton_schulz,
)

#: Section 3.1 / 4.2: ``gamma = 1``, ``r = sqrt(mean(cosh(x) ** 2))``.
VANILLA = {"temperature_alpha": 0.0, "brake_exponent": 0.5}
#: Section 3.3 eq. (5) / 4.2: ``gamma = N ** 0.48``, ``r = s ** (3.5 / 2)``.
TEMPERATURE_SCALED = {"temperature_alpha": 0.48, "brake_exponent": 1.75}


class AuON(Optimizer):
    """AuON: momentum followed by a linear-time cosh-RMS trust-region rescaling.

    Args:
        learning_rate: step size (the paper uses 0.24 for temperature-scaled
            AuON on language modelling; the damped update means this is not
            comparable to Muon's 0.01 -- see the README).
        momentum: EMA decay of the momentum buffer.
        weight_decay: decoupled weight decay.
        rho: spectral trust-region radius.  The update satisfies
            ``||dW||_2 < rho`` for every input.
        temperature_alpha: exponent of the cosh temperature ``N ** alpha``.
        brake_exponent: exponent applied to ``mean(cosh(...) ** 2)``.
        gamma: explicit temperature; overrides ``temperature_alpha``.
        eps0: guard on the Frobenius normalisation.
        eps: guard on the brake denominator.
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]] = 0.24,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        rho: float = 1.0,
        temperature_alpha: float = 0.0,
        brake_exponent: float = 0.5,
        gamma: float | None = None,
        eps0: float = EPS0,
        eps: float = EPS,
    ):
        super().__init__()
        if rho <= 0:
            raise ValueError(f"rho (trust-region radius) must be positive, got {rho}")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum must be in [0, 1], got {momentum}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
        self._maybe_schedule("learning_rate", learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.rho = rho
        self.temperature_alpha = temperature_alpha
        self.brake_exponent = brake_exponent
        self.gamma = gamma
        self.eps0 = eps0
        self.eps = eps

    def init_single(self, parameter: mx.array, state: dict):
        state["m"] = mx.zeros_like(parameter)

    def rescale(self, m: mx.array) -> tuple[mx.array, mx.array]:
        """The AuON rescaling of the momentum buffer (overridable by Hybrid)."""
        return cosh_rms_update(
            m,
            rho=self.rho,
            temperature_alpha=self.temperature_alpha,
            brake_exponent=self.brake_exponent,
            gamma=self.gamma,
            eps0=self.eps0,
            eps=self.eps,
        )

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        m = self.momentum * state["m"] + (1 - self.momentum) * gradient
        state["m"] = m
        update, _ = self.rescale(m)
        lr = self.learning_rate.astype(parameter.dtype)
        if self.weight_decay != 0:
            parameter = parameter * (1 - lr * self.weight_decay)
        return parameter - lr * update


class HybridAuON(AuON):
    """Hybrid-AuON: Newton-Schulz decorrelation, then the AuON brake.

    Only matrix (2-D) parameters are orthogonalised; the default ``ns_steps = 5``
    is the paper's ``Hybrid-AuON(NS=5)``, and ``ns_steps = 1`` is its ``AuON+``.
    This is the one place in the package with an iteration count, and it belongs
    to the Muon-style half of the method, not to AuON's rescaling.
    """

    def __init__(self, *args, ns_steps: int = 5, ns_coeffs=NS_SIMPLE, **kwargs):
        super().__init__(*args, **kwargs)
        if ns_steps < 1:
            raise ValueError(f"ns_steps must be at least 1, got {ns_steps}")
        self.ns_steps = ns_steps
        self.ns_coeffs = ns_coeffs

    def rescale(self, m: mx.array) -> tuple[mx.array, mx.array]:
        if m.ndim >= 2:
            m = newton_schulz(
                m, steps=self.ns_steps, eps0=self.eps0, coeffs=self.ns_coeffs
            )
        return super().rescale(m)


class Muon(Optimizer):
    """Muon: momentum followed by Newton-Schulz approximate orthogonalisation.

    The baseline this repo compares against, matching the standard
    implementation: ``ns_steps = 5`` Newton-Schulz iterations with the
    coefficients ``(3.4445, -4.7750, 2.0315)``, Nesterov momentum, decoupled
    weight decay and the shape-dependent scale ``sqrt(max(1, m/n))``.  The
    output is approximately semi-orthogonal, with spectral norm near 1 regardless
    of the gradient distribution.
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]] = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_coeffs=NS_SIMPLE,
        eps0: float = EPS0,
    ):
        super().__init__()
        self._maybe_schedule("learning_rate", learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.ns_coeffs = ns_coeffs
        self.eps0 = eps0

    def init_single(self, parameter: mx.array, state: dict):
        state["v"] = mx.zeros_like(parameter)

    def orthogonalize(self, update: mx.array) -> mx.array:
        """Newton-Schulz on a 2-D update; other shapes are left untouched."""
        if update.ndim < 2:
            return update
        return newton_schulz(
            update, steps=self.ns_steps, eps0=self.eps0, coeffs=self.ns_coeffs
        )

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        v = self.momentum * state["v"] + (1 - self.momentum) * gradient
        state["v"] = v
        if self.nesterov:
            update = gradient * (1 - self.momentum) + v * self.momentum
        else:
            update = v
        update = self.orthogonalize(update)
        lr = self.learning_rate.astype(parameter.dtype)
        if self.weight_decay != 0:
            parameter = parameter * (1 - lr * self.weight_decay)
        if update.ndim >= 2:
            lr = lr * muon_scale(update)
        return parameter - lr * update


#: Parameters the paper hands to Adam even when they are matrices.
ADAM_PARAM_MARKERS = ("tok_emb", "pos_emb", "head")


def is_hidden_matrix(path: str, weight: mx.array) -> bool:
    """True for the matrix parameters that AuON / Muon are responsible for."""
    return weight.ndim >= 2 and not any(
        marker in path for marker in ADAM_PARAM_MARKERS
    )


def split_optimizer(matrix_optimizer: Optimizer, other_optimizer: Optimizer):
    """Matrix parameters to ``matrix_optimizer``, everything else to the other.

    This is the split of the paper's section 4.2 setup: scalars, embeddings and
    the LM head are optimised with Adam, the hidden matrices with the method
    under test.
    """
    return MultiOptimizer([matrix_optimizer, other_optimizer], [is_hidden_matrix])


def build_optimizer(
    name: str,
    learning_rate: float,
    *,
    adam_learning_rate: float = 1e-3,
    momentum: float = 0.95,
    weight_decay: float = 0.0,
    rho: float = 1.0,
    ns_steps: int = 5,
    brake_exponent: float | None = None,
):
    """A ready-to-use optimizer for one of the arms of the experiment.

    All arms share the same Adam treatment of the non-hidden parameters; only
    the rule applied to the hidden matrices changes.
    """
    adam = Adam(learning_rate=adam_learning_rate)
    if name == "adam":
        return adam
    if name == "sgd_momentum":
        from mlx.optimizers import SGD

        matrix = SGD(learning_rate=learning_rate, momentum=momentum)
    elif name == "muon":
        matrix = Muon(
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
        )
    elif name == "auon":
        matrix = AuON(
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            rho=rho,
            **VANILLA,
        )
    elif name == "auon_temp":
        cfg = dict(TEMPERATURE_SCALED)
        if brake_exponent is not None:
            cfg["brake_exponent"] = brake_exponent
        matrix = AuON(
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            rho=rho,
            **cfg,
        )
    elif name == "hybrid_auon":
        cfg = dict(TEMPERATURE_SCALED)
        if brake_exponent is not None:
            cfg["brake_exponent"] = brake_exponent
        matrix = HybridAuON(
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            rho=rho,
            ns_steps=ns_steps,
            **cfg,
        )
    else:
        raise ValueError(f"unknown optimizer arm: {name!r}")
    return split_optimizer(matrix, adam)


def adam_parameters(model: nn.Module) -> list[str]:
    """Paths of the parameters that the split sends to Adam."""
    from mlx.utils import tree_flatten

    return [
        path
        for path, weight in tree_flatten(model.trainable_parameters())
        if not is_hidden_matrix(path, weight)
    ]
