"""AuON in MLX: a linear-time alternative to orthogonal momentum updates.

An independent, toy-scale implementation of AuON (arXiv:2509.24320).  The
package is built up in layers: :mod:`auon.normalizers` holds the cosh-RMS
rescaling and the Newton-Schulz iteration, :mod:`auon.optim` the optimizers,
and :mod:`auon.model` / :mod:`auon.data` / :mod:`auon.train` the experiment
harness.
"""

from .normalizers import (
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

__all__ = [
    "NS_CLASSIC",
    "NS_QUINTIC",
    "NS_SIMPLE",
    "cosh_rms",
    "cosh_rms_update",
    "muon_scale",
    "newton_schulz",
    "normalized_momentum",
    "spectral_norm",
]

__version__ = "0.1.0"
