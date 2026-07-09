"""Config wrapper for the CUDA closed-loop oracle hierarchy."""

from __future__ import annotations

from pathlib import Path

from blind_splitting_bddm.src.utils import common_parser, prepare_config, seed_all, write_markdown
from posterior_bddm_oracle.src.experiments_posterior_closed_loop_torch import run_experiment


def run(cfg: dict):
    seed = int(cfg.get("seed", 123))
    seed_all(seed)
    save_dir = Path(cfg.get("save_dir", "blind_splitting_bddm/results/closed_loop"))
    exp = cfg.get("experiments", {})
    prior = cfg.get("prior", {})
    meas = cfg.get("measurement", {})
    sigma_grid = cfg.get("sigma_grid", {})
    split_names = exp.get("split_methods", ["gradient"])
    payloads = []
    for split in split_names:
        out = save_dir / f"split_{split}"
        payloads.append(
            run_experiment(
                out=out,
                seed=seed,
                prior_name="ellipse" if prior.get("type", "ellipse") != "full" else "full",
                d_values=exp.get("d_values", [int(prior.get("ambient_dim", 50))]),
                measurement_ratios=exp.get("measurement_ratios", [float(meas.get("measurement_ratio", 0.5))]),
                noise_stds=exp.get("noise_std", [float(meas.get("noise_std", 0.05))]),
                eta_values=exp.get("etas", [0.003]),
                intrinsic_k=int(prior.get("intrinsic_dim", 2)),
                components=int(prior.get("n_components", 64)),
                n_trials=int(exp.get("n_trials", 64)),
                n_steps=int(exp.get("closed_loop_steps", 100)),
                grid_size=int(sigma_grid.get("n", 128)),
                sigma_min=float(sigma_grid.get("min", 0.01)),
                sigma_max=float(sigma_grid.get("max", 3.0)),
                sigma0=float(exp.get("sigma0", 1.2)),
                schedule_sigma_min=float(exp.get("schedule_sigma_min", 0.02)),
                init=exp.get("init", "posterior"),
                A_type="random",
                split_name=split,
                h=float(exp.get("h", 0.05)),
                beta=float(exp.get("beta", 0.0)),
                raw_hqs_tau=float(exp.get("raw_hqs_tau", 1e-2)),
                pdhg_gamma=float(exp.get("pdhg_gamma", exp.get("gammas", [100])[0])),
                hqs_tau=float(exp.get("hqs_tau", 1e-2)),
                device_name=cfg.get("device", "auto"),
                dtype_name=cfg.get("dtype", "float64"),
            )
        )
    lines = ["# Closed-Loop Hierarchy Wrapper", ""]
    lines.append(f"Ran {len(payloads)} split condition(s): {', '.join(split_names)}.")
    lines.append("Read each `split_*/data/closed_loop_cuda_report.md` for posterior hierarchy metrics.")
    save_dir.mkdir(parents=True, exist_ok=True)
    write_markdown(save_dir / "closed_loop_summary.md", lines)
    return payloads


def main() -> None:
    parser = common_parser("Run closed-loop hierarchy")
    args = parser.parse_args()
    run(prepare_config(args))


if __name__ == "__main__":
    main()
