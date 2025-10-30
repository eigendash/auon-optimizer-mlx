"""AuON in MLX: a linear-time alternative to orthogonal momentum updates.

An independent, toy-scale implementation of AuON (arXiv:2509.24320).  The
package is built up in layers: ``normalizers`` holds the cosh-RMS rescaling and
the Newton-Schulz iteration, ``optim`` the optimizers, ``model`` / ``data`` /
``train`` the experiment harness.
"""

__version__ = "0.1.0"
