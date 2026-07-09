"""Config wrapper for CUDA posterior-correction oracle tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from blind_splitting_bddm.src.utils import common_parser, prepare_config, seed_all, write_markdown
from posterior_bddm_oracle.src.experiments_posterior_oracle_torch import run_experiment


def _split_names(names: list[str]) -> list[str]:
    mapping = {"pg": "gradient", "gradient": "gradient", "pdhg": "pdhg", "hqs": "hqs", "admm": "hqs"}
    return [mapping[name] for name in names if name in mapping]


def run(cfg: dict) -> list[dict]:
    seed = int(cfg.get("seed", 123))
    seed_all(seed)
    save_dir = Path(cfg.get("save_dir", "blind_splitting_bddm/results/posterior_correction"))
    exp = cfg.get("experiments", {})
    prior = cfg.get("prior", {})
    meas = cfg.get("measurement", {})
    sigma_grid = cfg.get("sigma_grid", {})
    gammas = exp.get("gammas", [100])
    payloads = []
    for gamma in gammas:
        out = save_dir / f"gamma_{float(gamma):g}"
        payloads.append(
            run_experiment(
                out=out,
                seed=seed,
                prior_name="ellipse" if prior.get("type", "ellipse") != "full" else "full",
                d_values=exp.get("d_values", [int(prior.get("ambient_dim", 50))]),
                measurement_ratios=exp.get("measurement_ratios", [float(meas.get("measurement_ratio", 0.5))]),
                noise_stds=exp.get("noise_std", [float(meas.get("noise_std", 0.05))]),
                intrinsic_k=int(prior.get("intrinsic_dim", 2)),
                components=int(prior.get("n_components", 64)),
                sigmas=exp.get("sigmas", [0.05, 0.1, 0.3, 1.0]),
                etas=exp.get("etas", [3e-4, 1e-3, 3e-3, 1e-2]),
                split_methods=_split_names(exp.get("split_methods", ["gradient", "pdhg"])),
                n_samples=int(exp.get("n_trials", 128)),
                grid_size=int(sigma_grid.get("n", 128)),
                sigma_min=float(sigma_grid.get("min", 0.01)),
                sigma_max=float(sigma_grid.get("max", 3.0)),
                A_type="random",
                pdhg_gamma=float(gamma),
                hqs_tau=float(exp.get("hqs_tau", 1e-2)),
                device_name=cfg.get("device", "auto"),
                dtype_name=cfg.get("dtype", "float64"),
            )
        )
    lines = ["# Posterior Correction Oracle Wrapper", ""]
    lines.append(f"Ran {len(payloads)} gamma condition(s): {', '.join(str(g) for g in gammas)}.")
    lines.append("Read each `gamma_*/data/posterior_correction_cuda_report.md` for the correction-alignment table.")
    save_dir.mkdir(parents=True, exist_ok=True)
    write_markdown(save_dir / "posterior_correction_summary.md", lines)
    return payloads


def main() -> None:
    parser = common_parser("Run posterior correction oracle tests")
    args = parser.parse_args()
    run(prepare_config(args))


if __name__ == "__main__":
    main()
