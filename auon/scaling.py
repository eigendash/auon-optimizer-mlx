"""Cost of one optimizer step as a function of matrix size.

The linear-time claim is about the *shape* of the cost curve, so this module
measures wall-clock seconds per optimizer step on a single square matrix of
increasing side, and fits the exponent of ``t ~ n ** slope``.  AuON should give
``slope ~ 2`` (cost linear in the number of entries ``n ** 2``); Muon runs
``3 * ns_steps`` matrix products and should give ``slope ~ 3``.

The analytic column is a multiply-accumulate *proxy*, not a hardware
measurement: elementwise passes over the matrix for AuON, three matrix products
per Newton-Schulz iteration for Muon.
"""

from __future__ import annotations

import time
from typing import Callable

import mlx.core as mx
from mlx.optimizers import SGD, Adam, Optimizer

from .optim import TEMPERATURE_SCALED, VANILLA, AuON, HybridAuON, Muon


def optimizer_factory(name: str, learning_rate: float = 1e-2) -> Callable[[], Optimizer]:
    """A zero-argument constructor for the timing sweep."""
    if name == "adam":
        return lambda: Adam(learning_rate=learning_rate)
    if name == "sgd_momentum":
        return lambda: SGD(learning_rate=learning_rate, momentum=0.9)
    if name == "muon":
        return lambda: Muon(learning_rate=learning_rate, ns_steps=5)
    if name == "muon_ns1":
        return lambda: Muon(learning_rate=learning_rate, ns_steps=1)
    if name == "auon":
        return lambda: AuON(learning_rate=learning_rate, **VANILLA)
    if name == "auon_temp":
        return lambda: AuON(learning_rate=learning_rate, **TEMPERATURE_SCALED)
    if name == "hybrid_auon":
        return lambda: HybridAuON(learning_rate=learning_rate, ns_steps=5)
    if name == "hybrid_auon_ns1":
        return lambda: HybridAuON(learning_rate=learning_rate, ns_steps=1)
    raise ValueError(f"unknown optimizer for the scaling sweep: {name!r}")


def analytic_macs(name: str, n: int, ns_steps: int = 5) -> float:
    """Multiply-accumulate proxy for one step on an ``n x n`` matrix.

    AuON: five elementwise passes (Frobenius accumulation, normalisation,
    temperature, ``cosh`` square + mean, final division) over ``N = n**2``
    entries.  Muon: ``3 * ns_steps`` matrix products of size ``n`` (``X X^T``,
    ``A A``, ``B X``) plus the normalisation passes.
    """
    total = n * n
    if name in ("adam", "sgd_momentum"):
        return 4.0 * total
    if name in ("auon", "auon_temp"):
        return 5.0 * total
    if name in ("muon", "hybrid_auon"):
        return 3.0 * ns_steps * n**3 + 4.0 * total
    if name in ("muon_ns1", "hybrid_auon_ns1"):
        return 3.0 * 1 * n**3 + 4.0 * total
    raise ValueError(f"unknown optimizer: {name!r}")


def time_optimizer_step(
    optimizer: Optimizer,
    n: int,
    *,
    reps: int = 5,
    batches: int = 5,
    warmup: int = 3,
    seed: int = 0,
) -> float:
    """Fastest observed seconds per ``apply_gradients`` call on one ``n x n`` matrix.

    The minimum over batches is used rather than the mean: timing on a shared
    machine is one-sided noise, so the smallest sample is the closest to the
    real cost.
    """
    mx.random.seed(seed)
    weight = mx.random.normal(shape=(n, n))
    gradient = mx.random.normal(shape=(n, n)) * 0.1
    mx.eval(weight, gradient)

    def one_step():
        updated = optimizer.apply_gradients({"w": gradient}, {"w": weight})
        mx.eval(updated)

    for _ in range(warmup):
        one_step()
    samples = []
    for _ in range(batches):
        start = time.perf_counter()
        for _ in range(reps):
            one_step()
        samples.append((time.perf_counter() - start) / reps)
    return min(samples)


def loglog_slope(sizes: list[int], times: list[float]) -> float:
    """Least-squares exponent of ``times ~ sizes ** slope``."""
    xs = [mx.log(mx.array(float(s))) for s in sizes]
    ys = [mx.log(mx.array(float(t))) for t in times]
    xs = mx.stack(xs)
    ys = mx.stack(ys)
    slope = mx.sum((xs - xs.mean()) * (ys - ys.mean())) / mx.sum((xs - xs.mean()) ** 2)
    mx.eval(slope)
    return float(slope)


def scaling_sweep(
    names: list[str],
    sizes: list[int],
    *,
    reps: int = 5,
    batches: int = 3,
    progress=None,
) -> dict:
    """Time every optimizer at every size and fit the cost exponent."""
    results: dict = {"sizes": sizes, "arms": {}}
    for name in names:
        times, macs = [], []
        for n in sizes:
            optimizer = optimizer_factory(name)()
            seconds = time_optimizer_step(optimizer, n, reps=reps, batches=batches)
            times.append(seconds)
            macs.append(analytic_macs(name, n))
            if progress:
                progress(f"  {name:16s} n={n:5d}  {seconds * 1e3:9.3f} ms/step")
        results["arms"][name] = {
            "seconds_per_step": times,
            "ms_per_step": [t * 1e3 for t in times],
            "analytic_macs": macs,
            "slope": loglog_slope(sizes, times),
        }
    return results
