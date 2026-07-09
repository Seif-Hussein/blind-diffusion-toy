"""Wrappers around the CUDA closed-loop hierarchy runner."""

from __future__ import annotations

from posterior_bddm_oracle.src.experiments_posterior_closed_loop_torch import run_experiment

__all__ = ["run_experiment"]
