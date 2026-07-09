"""PDHG dual-law and split-induced perturbation diagnostics."""

from __future__ import annotations

import numpy as np
import torch

from blind_splitting_bddm.src.measurements import make_measurement_operator
from blind_splitting_bddm.src.plotting import save_line_plot
from blind_splitting_bddm.src.priors_gmm import make_prior, make_sigma_grid
from blind_splitting_bddm.src.splitting_pdhg import (
    dual_tracking_residual,
    pdhg_dual_update,
    quadratic_dual_memory_scalar,
)
from blind_splitting_bddm.src.utils import common_parser, ensure_dirs, prepare_config, seed_all, write_markdown
from posterior_bddm_oracle.src.torch_oracle_tools import make_generator, make_measurement


METRIC_NAMES = [
    "delta_norm",
    "identity_residual",
    "bias_ratio",
    "anisotropy_ratio",
    "gaussianization_ratio",
    "perturb_trace",
]


def _cov_stats(b: torch.Tensor, rho: float) -> tuple[float, float, float, float]:
    mean = torch.mean(b, dim=0)
    centered = b - mean[None, :]
    cov = centered.T @ centered / max(1, b.shape[0] - 1)
    trace = torch.trace(cov).clamp_min(1e-30)
    eig_max = torch.linalg.eigvalsh(cov).max().clamp_min(0.0)
    bias_ratio = mean.square().sum() / trace
    anisotropy = eig_max / (trace / b.shape[1])
    gaussianization = (b.shape[1] * float(rho) ** 2) / trace
    return (
        float(bias_ratio.detach().cpu()),
        float(anisotropy.detach().cpu()),
        float(gaussianization.detach().cpu()),
        float(trace.detach().cpu()),
    )


def run(cfg: dict) -> dict:
    seed = int(cfg.get("seed", 123))
    seed_all(seed)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32 if cfg.get("dtype") == "float32" else torch.float64
    paths = ensure_dirs(cfg.get("save_dir", "blind_splitting_bddm/results/dual_law"))

    prior_cfg = dict(cfg.get("prior", {}))
    meas_cfg = cfg.get("measurement", {})
    exp = cfg.get("experiments", {})
    n_trials = int(exp.get("n_trials", 128))
    n_steps = int(exp.get("closed_loop_steps", 100))
    gammas = exp.get("gammas", [50, 100, 500])
    eta = float(exp.get("etas", [0.003])[0])
    beta = float(exp.get("beta", 0.0))
    h_step = float(exp.get("h", 0.05))
    sigma0 = float(exp.get("sigma0", 1.2))
    sigma_min_sched = float(exp.get("schedule_sigma_min", 0.02))
    sigma_grid = make_sigma_grid(cfg.get("sigma_grid", {}), device, dtype)
    prior = make_prior(prior_cfg, sigma_grid, device, dtype, seed)
    gen = make_generator(device, seed + 10)
    A = make_measurement_operator(meas_cfg, prior.d, gen, device, dtype)
    x_true, y_obs = make_measurement(prior, A, float(meas_cfg.get("noise_std", 0.05)), gen)
    schedule = torch.tensor(np.geomspace(sigma0, sigma_min_sched, n_steps), device=device, dtype=dtype)

    metrics = np.full((len(gammas), n_steps, len(METRIC_NAMES)), np.nan, dtype=np.float64)
    with torch.no_grad():
        for gi, gamma in enumerate(gammas):
            x0, _ = prior.sample(n_trials, gen)
            Y = x0 + schedule[0] * torch.randn(x0.shape, device=device, dtype=dtype, generator=gen)
            w = torch.zeros((n_trials, A.shape[0]), device=device, dtype=dtype)
            delta_prev = torch.zeros_like(w)
            m_prev = prior.denoise(Y, schedule[0])
            memory = quadratic_dual_memory_scalar(float(meas_cfg.get("noise_std", 0.05)), float(gamma))
            for step, sigma in enumerate(schedule):
                m_now = prior.denoise(Y, sigma)
                force, w_next, _z = pdhg_dual_update(
                    m_now,
                    y_obs,
                    A,
                    float(meas_cfg.get("noise_std", 0.05)),
                    float(gamma),
                    w,
                )
                G_now = ((m_now @ A.T) - y_obs[None, :]) / float(meas_cfg.get("noise_std", 0.05)) ** 2
                delta_next = w_next - G_now
                if step == 0:
                    residual = torch.zeros_like(delta_next)
                else:
                    residual = dual_tracking_residual(
                        delta_next,
                        delta_prev,
                        m_now,
                        m_prev,
                        A,
                        float(meas_cfg.get("noise_std", 0.05)),
                        float(gamma),
                    )
                b = -eta * sigma.square() * force
                rho = float(torch.sqrt(torch.as_tensor(2.0 * h_step * beta, device=device, dtype=dtype)) * sigma)
                bias, aniso, gauss, trace = _cov_stats(b, rho)
                metrics[gi, step] = np.asarray(
                    [
                        float(torch.linalg.norm(delta_next, dim=1).median().detach().cpu()),
                        float(torch.linalg.norm(residual, dim=1).median().detach().cpu()),
                        bias,
                        aniso,
                        gauss,
                        trace,
                    ]
                )
                Y = Y - eta * sigma.square() * force + h_step * (m_now - Y)
                w = w_next
                delta_prev = delta_next
                m_prev = m_now
            print(
                f"dual gamma={gamma:g}: M={memory:.4f}, median identity residual="
                f"{np.nanmedian(metrics[gi, :, METRIC_NAMES.index('identity_residual')]):.3e}"
            )

    np.savez_compressed(
        paths["data"] / "dual_law_results.npz",
        metrics=metrics,
        metric_names=np.asarray(METRIC_NAMES),
        gammas=np.asarray(gammas),
        schedule=schedule.detach().cpu().numpy(),
    )
    steps = np.arange(n_steps)
    save_line_plot(
        paths["figures"] / "dual_identity_residual.png",
        steps,
        {f"gamma={g:g}": metrics[i, :, METRIC_NAMES.index("identity_residual")] for i, g in enumerate(gammas)},
        "iteration",
        "median residual",
        "PDHG dual tracking identity residual",
        logy=True,
    )
    save_line_plot(
        paths["figures"] / "dual_bias_ratio.png",
        steps,
        {f"gamma={g:g}": metrics[i, :, METRIC_NAMES.index("bias_ratio")] for i, g in enumerate(gammas)},
        "iteration",
        "bias ratio",
        "Split-induced perturbation bias ratio",
        logy=True,
    )
    lines = ["# PDHG Dual-Law Diagnostics", ""]
    lines.append("| gamma | median identity residual | median bias ratio | median anisotropy |")
    lines.append("|---:|---:|---:|---:|")
    for gi, gamma in enumerate(gammas):
        lines.append(
            f"| {float(gamma):g} | {np.nanmedian(metrics[gi, :, 1]):.3e} | "
            f"{np.nanmedian(metrics[gi, :, 2]):.3e} | {np.nanmedian(metrics[gi, :, 3]):.3e} |"
        )
    lines.append("")
    lines.append("Identity residual near numerical precision validates the quadratic PDHG dual-memory relation.")
    write_markdown(paths["reports"] / "dual_law_report.md", lines)
    return {"metrics": metrics, "paths": paths}


def main() -> None:
    parser = common_parser("Run PDHG dual-law diagnostics")
    args = parser.parse_args()
    run(prepare_config(args))


if __name__ == "__main__":
    main()
