"""CUDA-accelerated posterior correction oracle diagnostics.

This is the Colab/GPU port of the key one-step viability test:

    c_star = m_pi(Y, sigma) - m_p(Y, sigma)

against cheap splitting corrections in noisy coordinates.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .torch_oracle_tools import (
    cosine_similarity,
    exact_linear_posterior_gmm,
    make_device,
    make_ellipse_gmm_torch,
    make_full_gaussian_torch,
    make_generator,
    make_measurement,
    make_operator,
    parse_float_list,
    parse_int_list,
    parse_str_list,
    relative_error,
    split_force,
    sync,
    tangent_normal_fraction,
)


METRIC_NAMES = [
    "rel_error",
    "cosine",
    "norm_ratio",
    "gain_to_target",
    "gain_fit_rel_error",
    "c_star_norm",
    "c_split_norm",
    "c_star_normal_fraction",
    "c_split_normal_fraction",
]

SCALE_NAMES = [
    "sigma_hat_prior",
    "sigma_hat_posterior",
    "sigma_entropy_prior",
    "sigma_entropy_posterior",
    "prior_sigma_min_hit",
    "prior_sigma_max_hit",
    "posterior_sigma_min_hit",
    "posterior_sigma_max_hit",
]

CONDITION_NAMES = ["d", "intrinsic_k", "measurement_ratio", "m", "noise_std"]


def _condition_table(
    d_values: list[int],
    measurement_ratios: list[float],
    noise_stds: list[float],
    intrinsic_k: int,
) -> np.ndarray:
    rows = []
    for d in d_values:
        for ratio in measurement_ratios:
            m = max(1, int(round(float(ratio) * d)))
            for noise_std in noise_stds:
                rows.append((d, intrinsic_k, ratio, m, noise_std))
    return np.asarray(rows, dtype=float)


def _make_prior(
    prior_name: str,
    d: int,
    components: int,
    sigma_grid: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    if prior_name == "ellipse":
        return make_ellipse_gmm_torch(d, components, sigma_grid, device, dtype, seed)
    if prior_name == "full":
        return make_full_gaussian_torch(d, 1.0, sigma_grid, device, dtype)
    raise ValueError(f"unknown prior: {prior_name}")


def _write_report(
    path: Path,
    payload: dict,
) -> None:
    metrics = payload["metrics"]
    scale = payload["scale_metrics"]
    conditions = payload["conditions"]
    sigmas = payload["sigmas"]
    etas = payload["etas"]
    splits = [str(x) for x in payload["split_names"]]

    rel_idx = METRIC_NAMES.index("rel_error")
    cos_idx = METRIC_NAMES.index("cosine")
    p_min = SCALE_NAMES.index("prior_sigma_min_hit")
    p_max = SCALE_NAMES.index("prior_sigma_max_hit")
    pi_min = SCALE_NAMES.index("posterior_sigma_min_hit")
    pi_max = SCALE_NAMES.index("posterior_sigma_max_hit")

    lines = ["# CUDA Posterior Correction Oracle Report", ""]
    lines.append(f"Device: `{payload['device']}`")
    lines.append(f"Dtype: `{payload['dtype']}`")
    lines.append("")
    lines.append("## Aggregate Split Alignment")
    lines.append("| split | median rel error | median cosine | median norm ratio | median gain fit error |")
    lines.append("|---|---:|---:|---:|---:|")
    for mi, split in enumerate(splits):
        rel = np.nanmedian(metrics[:, :, :, mi, :, rel_idx])
        cos = np.nanmedian(metrics[:, :, :, mi, :, cos_idx])
        norm_ratio = np.nanmedian(metrics[:, :, :, mi, :, METRIC_NAMES.index("norm_ratio")])
        gain_fit = np.nanmedian(metrics[:, :, :, mi, :, METRIC_NAMES.index("gain_fit_rel_error")])
        lines.append(f"| `{split}` | {rel:.4e} | {cos:.4f} | {norm_ratio:.4e} | {gain_fit:.4e} |")
    lines.append("")

    lines.append("## Best Eta By Condition")
    lines.append(
        "| d | ratio | noise | sigma | split | best eta | median rel error | "
        "median cosine | median norm ratio | edge best |"
    )
    lines.append("|---:|---:|---:|---:|---|---:|---:|---:|---:|")
    for ci, cond in enumerate(conditions):
        d, _k, ratio, _m, noise_std = cond
        for si, sigma in enumerate(sigmas):
            for mi, split in enumerate(splits):
                rel_by_eta = np.nanmedian(metrics[ci, si, :, mi, :, rel_idx], axis=1)
                best = int(np.nanargmin(rel_by_eta))
                cos = np.nanmedian(metrics[ci, si, best, mi, :, cos_idx])
                norm_ratio = np.nanmedian(
                    metrics[ci, si, best, mi, :, METRIC_NAMES.index("norm_ratio")]
                )
                edge_best = best == 0 or best == len(etas) - 1
                lines.append(
                    f"| {int(d)} | {ratio:g} | {noise_std:g} | {sigma:g} | `{split}` | "
                    f"{etas[best]:g} | {rel_by_eta[best]:.4e} | {cos:.4f} | "
                    f"{norm_ratio:.4e} | {edge_best} |"
                )
    lines.append("")
    lines.append(
        "`edge best=True` means the best eta was on a sweep boundary, so the eta range is not yet resolving "
        "the optimum."
    )
    lines.append("")

    lines.append("## Scale Boundary Hits")
    lines.append("| d | ratio | noise | sigma | prior hit | posterior hit |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for ci, cond in enumerate(conditions):
        d, _k, ratio, _m, noise_std = cond
        for si, sigma in enumerate(sigmas):
            prior_hit = np.nanmean(scale[ci, si, :, p_min] + scale[ci, si, :, p_max])
            post_hit = np.nanmean(scale[ci, si, :, pi_min] + scale[ci, si, :, pi_max])
            lines.append(
                f"| {int(d)} | {ratio:g} | {noise_std:g} | {sigma:g} | "
                f"{prior_hit:.3f} | {post_hit:.3f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(
    *,
    out: str | Path,
    seed: int,
    prior_name: str,
    d_values: list[int],
    measurement_ratios: list[float],
    noise_stds: list[float],
    intrinsic_k: int,
    components: int,
    sigmas: list[float],
    etas: list[float],
    split_methods: list[str],
    n_samples: int,
    grid_size: int,
    sigma_min: float,
    sigma_max: float,
    A_type: str,
    pdhg_gamma: float,
    hqs_tau: float,
    device_name: str,
    dtype_name: str,
) -> dict:
    device = make_device(device_name)
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    out = Path(out)
    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    conditions = _condition_table(d_values, measurement_ratios, noise_stds, intrinsic_k)
    sigma_grid = torch.tensor(np.geomspace(sigma_min, sigma_max, grid_size), device=device, dtype=dtype)
    sigma_values = torch.tensor(sigmas, device=device, dtype=dtype)

    metrics = np.full(
        (
            conditions.shape[0],
            len(sigmas),
            len(etas),
            len(split_methods),
            n_samples,
            len(METRIC_NAMES),
        ),
        np.nan,
        dtype=np.float32,
    )
    scale_metrics = np.full(
        (conditions.shape[0], len(sigmas), n_samples, len(SCALE_NAMES)),
        np.nan,
        dtype=np.float32,
    )
    condition_times = np.full((conditions.shape[0],), np.nan, dtype=np.float32)

    total = conditions.shape[0]
    start_all = time.perf_counter()
    with torch.no_grad():
        for ci, cond in enumerate(conditions):
            d, _k, ratio, m, noise_std = cond
            d = int(d)
            m = int(m)
            noise_std = float(noise_std)
            start = time.perf_counter()
            prior = _make_prior(
                prior_name,
                d,
                components,
                sigma_grid,
                device,
                dtype,
                seed + 1000 * ci,
            )
            gen = make_generator(device, seed + 100000 * ci)
            A = make_operator(d, m, A_type, gen, device, dtype)
            _x_true, y_obs = make_measurement(prior, A, noise_std, gen)
            posterior_prior, _posterior = exact_linear_posterior_gmm(prior, A, y_obs, noise_std)
            sync(device)

            print(
                f"condition {ci + 1}/{total}: d={d}, ratio={ratio:g}, m={m}, "
                f"noise={noise_std:g}, samples={n_samples}, device={device}",
                flush=True,
            )
            for si, sigma_t in enumerate(sigma_values):
                x, _ids = posterior_prior.sample(n_samples, gen)
                Y = x + sigma_t * torch.randn(x.shape, device=device, dtype=dtype, generator=gen)
                m_prior = prior.denoise(Y, sigma_t)
                m_post = posterior_prior.denoise(Y, sigma_t)
                c_star = m_post - m_prior

                sigma_hat_prior = prior.sigma_mle(Y)
                sigma_hat_post = posterior_prior.sigma_mle(Y)
                scale_metrics[ci, si, :, SCALE_NAMES.index("sigma_hat_prior")] = (
                    sigma_hat_prior.detach().cpu().numpy().astype(np.float32)
                )
                scale_metrics[ci, si, :, SCALE_NAMES.index("sigma_hat_posterior")] = (
                    sigma_hat_post.detach().cpu().numpy().astype(np.float32)
                )
                scale_metrics[ci, si, :, SCALE_NAMES.index("sigma_entropy_prior")] = (
                    prior.sigma_entropy(Y).detach().cpu().numpy().astype(np.float32)
                )
                scale_metrics[ci, si, :, SCALE_NAMES.index("sigma_entropy_posterior")] = (
                    posterior_prior.sigma_entropy(Y).detach().cpu().numpy().astype(np.float32)
                )
                scale_metrics[ci, si, :, SCALE_NAMES.index("prior_sigma_min_hit")] = (
                    (sigma_hat_prior <= sigma_grid[0] * 1.000001).to(dtype).detach().cpu().numpy()
                )
                scale_metrics[ci, si, :, SCALE_NAMES.index("prior_sigma_max_hit")] = (
                    (sigma_hat_prior >= sigma_grid[-1] * 0.999999).to(dtype).detach().cpu().numpy()
                )
                scale_metrics[ci, si, :, SCALE_NAMES.index("posterior_sigma_min_hit")] = (
                    (sigma_hat_post <= sigma_grid[0] * 1.000001).to(dtype).detach().cpu().numpy()
                )
                scale_metrics[ci, si, :, SCALE_NAMES.index("posterior_sigma_max_hit")] = (
                    (sigma_hat_post >= sigma_grid[-1] * 0.999999).to(dtype).detach().cpu().numpy()
                )

                c_star_normal = tangent_normal_fraction(prior, m_prior, c_star)
                for ei, eta in enumerate(etas):
                    eta_f = float(eta)
                    for mi, method in enumerate(split_methods):
                        force = split_force(method, m_prior, y_obs, A, noise_std, pdhg_gamma, hqs_tau)
                        c_split = prior.denoise(Y - eta_f * sigma_t.square() * force, sigma_t) - m_prior
                        metrics[ci, si, ei, mi, :, METRIC_NAMES.index("rel_error")] = (
                            relative_error(c_split, c_star).detach().cpu().numpy().astype(np.float32)
                        )
                        metrics[ci, si, ei, mi, :, METRIC_NAMES.index("cosine")] = (
                            cosine_similarity(c_split, c_star).detach().cpu().numpy().astype(np.float32)
                        )
                        c_split_norm = torch.linalg.norm(c_split, dim=1)
                        c_star_norm = torch.linalg.norm(c_star, dim=1)
                        dot = torch.sum(c_split * c_star, dim=1)
                        gain_to_target = dot / c_split_norm.square().clamp_min(1e-30)
                        gain_fit = torch.linalg.norm(
                            gain_to_target[:, None] * c_split - c_star,
                            dim=1,
                        ) / c_star_norm.clamp_min(1e-15)
                        metrics[ci, si, ei, mi, :, METRIC_NAMES.index("norm_ratio")] = (
                            (c_split_norm / c_star_norm.clamp_min(1e-15))
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                        metrics[ci, si, ei, mi, :, METRIC_NAMES.index("gain_to_target")] = (
                            gain_to_target.detach().cpu().numpy().astype(np.float32)
                        )
                        metrics[ci, si, ei, mi, :, METRIC_NAMES.index("gain_fit_rel_error")] = (
                            gain_fit.detach().cpu().numpy().astype(np.float32)
                        )
                        metrics[ci, si, ei, mi, :, METRIC_NAMES.index("c_star_norm")] = (
                            c_star_norm.detach().cpu().numpy().astype(np.float32)
                        )
                        metrics[ci, si, ei, mi, :, METRIC_NAMES.index("c_split_norm")] = (
                            c_split_norm.detach().cpu().numpy().astype(np.float32)
                        )
                        metrics[ci, si, ei, mi, :, METRIC_NAMES.index("c_star_normal_fraction")] = (
                            c_star_normal.detach().cpu().numpy().astype(np.float32)
                        )
                        metrics[ci, si, ei, mi, :, METRIC_NAMES.index("c_split_normal_fraction")] = (
                            tangent_normal_fraction(prior, m_prior, c_split)
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                sync(device)

            condition_times[ci] = time.perf_counter() - start
            elapsed = time.perf_counter() - start_all
            avg = elapsed / (ci + 1)
            remaining = avg * (total - ci - 1)
            print(
                f"  condition_time={condition_times[ci]:.1f}s; eta_remaining={remaining / 60.0:.1f}m",
                flush=True,
            )
            np.savez_compressed(
                data_dir / "posterior_correction_cuda_partial.npz",
                metrics=metrics,
                metric_names=np.asarray(METRIC_NAMES),
                scale_metrics=scale_metrics,
                scale_names=np.asarray(SCALE_NAMES),
                conditions=conditions,
                condition_names=np.asarray(CONDITION_NAMES),
                sigmas=np.asarray(sigmas),
                etas=np.asarray(etas),
                split_names=np.asarray(split_methods),
                condition_times=condition_times,
                device=np.asarray(str(device)),
                dtype=np.asarray(dtype_name),
            )

    payload = {
        "metrics": metrics,
        "metric_names": np.asarray(METRIC_NAMES),
        "scale_metrics": scale_metrics,
        "scale_names": np.asarray(SCALE_NAMES),
        "conditions": conditions,
        "condition_names": np.asarray(CONDITION_NAMES),
        "sigmas": np.asarray(sigmas),
        "etas": np.asarray(etas),
        "split_names": np.asarray(split_methods),
        "condition_times": condition_times,
        "device": np.asarray(str(device)),
        "dtype": np.asarray(dtype_name),
    }
    np.savez_compressed(data_dir / "posterior_correction_cuda.npz", **payload)
    config = {
        "seed": seed,
        "prior": prior_name,
        "d_values": d_values,
        "measurement_ratios": measurement_ratios,
        "noise_stds": noise_stds,
        "intrinsic_k": intrinsic_k,
        "components": components,
        "sigmas": sigmas,
        "etas": etas,
        "split_methods": split_methods,
        "n_samples": n_samples,
        "grid_size": grid_size,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "A_type": A_type,
        "pdhg_gamma": pdhg_gamma,
        "hqs_tau": hqs_tau,
        "device": str(device),
        "dtype": dtype_name,
    }
    (data_dir / "posterior_correction_cuda_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    _write_report(data_dir / "posterior_correction_cuda_report.md", payload)
    print(f"saved CUDA posterior correction diagnostics to {data_dir / 'posterior_correction_cuda.npz'}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="posterior_bddm_oracle/results_cuda")
    parser.add_argument("--seed", type=int, default=202)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--prior", choices=["ellipse", "full"], default="ellipse")
    parser.add_argument("--d-values", default="2,10,50,100,500")
    parser.add_argument("--measurement-ratios", default="0.5")
    parser.add_argument("--noise-stds", default="0.08")
    parser.add_argument("--intrinsic-k", type=int, default=2)
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--sigmas", default="0.03,0.1,0.3,1.0")
    parser.add_argument("--eta-values", default="1e-4,3e-4,1e-3,3e-3,1e-2,3e-2")
    parser.add_argument("--split-methods", default="gradient,pdhg,hqs")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--grid-size", type=int, default=61)
    parser.add_argument("--sigma-min", type=float, default=0.005)
    parser.add_argument("--sigma-max", type=float, default=3.0)
    parser.add_argument("--A-type", choices=["random", "mask"], default="random")
    parser.add_argument("--pdhg-gamma", type=float, default=100.0)
    parser.add_argument("--hqs-tau", type=float, default=1e-2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = parser.parse_args()

    if args.quick:
        args.d_values = "2,10"
        args.sigmas = "0.1,0.3"
        args.eta_values = "1e-3,1e-2"
        args.n_samples = min(args.n_samples, 32)
        args.components = min(args.components, 8)
        args.grid_size = min(args.grid_size, 25)

    run_experiment(
        out=args.out,
        seed=args.seed,
        prior_name=args.prior,
        d_values=parse_int_list(args.d_values),
        measurement_ratios=parse_float_list(args.measurement_ratios),
        noise_stds=parse_float_list(args.noise_stds),
        intrinsic_k=args.intrinsic_k,
        components=args.components,
        sigmas=parse_float_list(args.sigmas),
        etas=parse_float_list(args.eta_values),
        split_methods=parse_str_list(args.split_methods),
        n_samples=args.n_samples,
        grid_size=args.grid_size,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        A_type=args.A_type,
        pdhg_gamma=args.pdhg_gamma,
        hqs_tau=args.hqs_tau,
        device_name=args.device,
        dtype_name=args.dtype,
    )


if __name__ == "__main__":
    main()
