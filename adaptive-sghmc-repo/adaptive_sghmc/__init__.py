"""Adaptive SGHMC experiment toolkit.

This repository packages a practical epsilon-parameterized adaptive SGHMC
sampler together with small toy examples and a Yacht regression benchmark
pipeline.
"""

from .samplers import samples_eps

__all__ = ["samples_eps"]
