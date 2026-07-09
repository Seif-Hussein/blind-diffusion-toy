"""Small plotting helpers for reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def save_line_plot(path: str | Path, x, ys: dict[str, np.ndarray], xlabel: str, ylabel: str, title: str, logx=False, logy=False):
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for label, y in ys.items():
        ax.plot(x, y, marker="o", label=label)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
