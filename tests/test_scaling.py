"""Cost model: the analytic op counts and the measured shape of the time curve.

The deterministic part (the multiply-accumulate proxy and the slope fit) carries
the claim; the measured ratios are a sanity check with deliberately loose bounds
because they run on a shared CPU.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from auon.scaling import (
    analytic_macs,
    loglog_slope,
    optimizer_factory,
    scaling_sweep,
    time_optimizer_step,
)

mx.random.seed(0)


def test_analytic_macs_is_linear_in_entries_for_auon():
    for n in (64, 256, 1024):
        entries = n * n
        assert analytic_macs("auon", n) == pytest.approx(5.0 * entries)
        assert analytic_macs("auon_temp", n) == pytest.approx(5.0 * entries)
        # doubling the side multiplies the work by four
        assert analytic_macs("auon", 2 * n) / analytic_macs("auon", n) == pytest.approx(4.0)


def test_analytic_macs_is_cubic_for_newton_schulz_arms():
    for n in (512, 1024):
        # three matrix products per Newton-Schulz iteration
        assert analytic_macs("muon", n, ns_steps=5) >= 15.0 * n**3
        assert analytic_macs("hybrid_auon", n, ns_steps=5) >= 15.0 * n**3
        assert analytic_macs("muon_ns1", n, ns_steps=1) >= 3.0 * n**3
    assert analytic_macs("muon", 1024) / analytic_macs("muon", 512) == pytest.approx(8.0, rel=0.05)
    # and the gap to AuON keeps widening with size
    gap_small = analytic_macs("muon", 64) / analytic_macs("auon", 64)
    gap_large = analytic_macs("muon", 1024) / analytic_macs("auon", 1024)
    # the gap grows like n, i.e. by the factor the side grew
    assert gap_large > 10 * gap_small


def test_loglog_slope_recovers_a_known_exponent():
    sizes = [1, 2, 4, 8, 16]
    assert loglog_slope(sizes, [float(n) ** 2 for n in sizes]) == pytest.approx(2.0, abs=1e-4)
    assert loglog_slope(sizes, [float(n) ** 3 for n in sizes]) == pytest.approx(3.0, abs=1e-4)
    assert loglog_slope(sizes, [float(n) ** 1.5 for n in sizes]) == pytest.approx(1.5, abs=1e-4)


def test_optimizer_factory_builds_every_documented_arm():
    names = [
        "adam",
        "sgd_momentum",
        "auon",
        "auon_temp",
        "hybrid_auon",
        "hybrid_auon_ns1",
        "muon",
        "muon_ns1",
    ]
    for name in names:
        optimizer = optimizer_factory(name)()
        weight = mx.zeros((8, 8))
        updated = optimizer.apply_gradients({"w": mx.ones((8, 8))}, {"w": weight})
        mx.eval(updated["w"])
        assert bool(mx.all(mx.isfinite(updated["w"]))), name
    with pytest.raises(ValueError):
        optimizer_factory("bogus")


def test_measured_time_grows_much_faster_for_muon_than_for_auon():
    """Loose timing bounds: 4x the side (16x the entries) must cost AuON far
    less than 16x and Muon more than that.  The exact numbers are in results/."""
    small, large = 256, 1024
    times = {}
    for name in ("auon", "muon"):
        times[name] = [
            time_optimizer_step(optimizer_factory(name)(), n, reps=5, batches=5)
            for n in (small, large)
        ]
    growth_auon = times["auon"][1] / times["auon"][0]
    growth_muon = times["muon"][1] / times["muon"][0]
    assert growth_auon < 40, growth_auon
    assert growth_muon > 12, growth_muon
    assert growth_muon > growth_auon


def test_scaling_sweep_returns_a_consistent_table():
    sizes = [32, 64]
    sweep = scaling_sweep(["auon", "muon"], sizes, reps=2, batches=2)
    assert sweep["sizes"] == sizes
    assert set(sweep["arms"]) == {"auon", "muon"}
    for name, arm in sweep["arms"].items():
        assert len(arm["seconds_per_step"]) == len(sizes)
        assert len(arm["analytic_macs"]) == len(sizes)
        assert all(t > 0 for t in arm["seconds_per_step"])
        assert np.isfinite(arm["slope"])
