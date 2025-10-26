"""Experiment modules."""
from __future__ import annotations

# Import experiments so they register themselves with the registry on module load.
from . import scientsbank  # noqa: F401

__all__ = ["scientsbank"]

