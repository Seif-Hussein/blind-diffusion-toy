"""Linear Gaussian measurement models."""

from __future__ import annotations

import torch

from posterior_bddm_oracle.src.torch_oracle_tools import data_gradient, make_measurement, make_operator


def make_measurement_operator(cfg: dict, d: int, gen: torch.Generator, device: torch.device, dtype: torch.dtype):
    ratio = float(cfg.get("measurement_ratio", 0.5))
    m = int(cfg.get("m", max(1, round(ratio * d))))
    kind = cfg.get("type", "random_gaussian")
    if kind == "random_gaussian":
        kind = "random"
    if kind in ("coordinate_selector", "coordinate_mask", "mask"):
        kind = "mask"
    return make_operator(d, m, kind, gen, device, dtype)


__all__ = ["data_gradient", "make_measurement", "make_measurement_operator"]
