"""Shared pytest configuration.

The whole repository is run on MLX's CPU device.  Training with these models is
bit-reproducible there, and a few operations used in the tests (``mx.linalg.svd``)
are only implemented on the CPU backend.
"""

from __future__ import annotations

import mlx.core as mx

mx.set_default_device(mx.cpu)
