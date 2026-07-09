"""One-step mechanism tests for closed-form blind diffusion splitting."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .metrics import local_tangent_normal_ratio
from .plotting import (
    ensure_dir,
    plot_likelihood_identity,
    plot_mean_std_vs_ratio,
    plot_scale_histograms,
    plot_tangent_filtering,
)
from .priors import (
    GaussianMixturePrior,
    make_ellipse_gmm,
    make_full_gaussian_control,
    make_subspace_gmm,
)


def _parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _make_prior(
    name: str,
    d: int,
    rng: np.random.Generator,
    sigma_grid: np.ndarray,
    curve_components: int,
) -> GaussianMixturePrior:
    if name == "subspace":
        return make_subspace_gmm(d, n_components=min(8, curve_components), rng=rng, sigma_grid=sigma_grid)
    if name == "ellipse":
        return make_ellipse_gmm(d, n_components=curve_components, rng=rng, sigma_grid=sigma_grid)
    if name == "full":
        return make_full_gaussian_control(d, rng=rng, sigma_grid=sigma_grid)
    raise ValueError(f"unknown prior: {name}")


def run_scale_concentration(args, figure_dir: Path) -> dict:
    rng = np.random.default_rng(args.seed)
    sigma_grid = np.geomspace(args.sigma_min, args.sigma_max, args.grid_size)
    d_values = _parse_int_list(args.d_values)
    results = {}
    for prior_name in args.priors:
        ratio_sq = []
        ratio = []
        intrinsic_k = 2
        for d in d_values:
            prior = _make_prior(prior_name, d, rng, sigma_grid, args.curve_components)
            intrinsic_k = int(prior.metadata.get("intrinsic_k", d))
            x, _ = prior.sample(args.n_samples, rng)
            y = x + args.true_sigma * rng.normal(size=x.shape)
            sigma_hat = np.asarray(prior.sigma_mle(y), dtype=float)
            ratio_sq.append((sigma_hat / args.true_sigma) ** 2)
            ratio.append(sigma_hat / args.true_sigma)
        ratio_sq_arr = np.vstack(ratio_sq)
        ratio_arr = np.vstack(ratio)
        plot_scale_histograms(
            d_values,
            ratio_sq_arr,
            figure_dir / f"scale_hist_{prior_name}.png",
            f"Blind scale concentration: {prior_name}",
        )
        plot_mean_std_vs_ratio(
            d_values,
            intrinsic_k,
            np.mean(ratio_arr, axis=1),
            np.std(ratio_arr, axis=1),
            figure_dir / f"scale_mean_std_{prior_name}.png",
            f"Blind scale ratio: {prior_name}",
        )
        results[prior_name] = {
            "ratio_sq": ratio_sq_arr,
            "ratio": ratio_arr,
            "intrinsic_k": intrinsic_k,
        }
    return {"d_values": np.asarray(d_values), "scale": results}


def run_likelihood_identity(args, figure_dir: Path) -> dict:
    rng = np.random.default_rng(args.seed + 101)
    sigma_grid = np.geomspace(args.sigma_min, args.sigma_max, args.grid_size)
    prior = make_ellipse_gmm(
        args.identity_d,
        n_components=args.curve_components,
        rng=rng,
        sigma_grid=sigma_grid,
    )
    x, _ = prior.sample(args.n_identity_samples, rng)
    y = x + args.identity_sigma * rng.normal(size=x.shape)
    g = rng.normal(size=x.shape)
    g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-15)

    etas = np.asarray(args.identity_etas, dtype=float)
    rel_errors = np.empty((etas.size, args.n_identity_samples), dtype=float)
    for e, eta in enumerate(etas):
        for n in range(args.n_identity_samples):
            m0 = prior.denoise(y[n], args.identity_sigma)
            m1 = prior.denoise(y[n] - eta * args.identity_sigma**2 * g[n], args.identity_sigma)
            C = prior.posterior_covariance(y[n], args.identity_sigma)
            delta_cov = -eta * (C @ g[n])
            rel_errors[e, n] = np.linalg.norm((m1 - m0) - delta_cov) / max(
                np.linalg.norm(delta_cov), 1e-15
            )
    plot_likelihood_identity(
        etas,
        np.mean(rel_errors, axis=1),
        np.std(rel_errors, axis=1),
        figure_dir / "likelihood_tilt_identity.png",
    )
    return {"etas": etas, "rel_errors": rel_errors}


def run_tangent_filter(args, figure_dir: Path) -> dict:
    rng = np.random.default_rng(args.seed + 202)
    sigma_grid = np.geomspace(args.sigma_min, args.sigma_max, args.grid_size)
    prior = make_ellipse_gmm(
        args.identity_d,
        n_components=args.curve_components,
        rng=rng,
        sigma_grid=sigma_grid,
    )
    x, _ = prior.sample(args.n_identity_samples, rng)
    y = x + args.identity_sigma * rng.normal(size=x.shape)
    g = rng.normal(size=x.shape)
    raw_ratio = np.empty(args.n_identity_samples, dtype=float)
    cov_ratio = np.empty(args.n_identity_samples, dtype=float)
    denoise_ratio = np.empty(args.n_identity_samples, dtype=float)
    for n in range(args.n_identity_samples):
        m = prior.denoise(y[n], args.identity_sigma)
        C = prior.posterior_covariance(y[n], args.identity_sigma)
        cov_g = C @ g[n]
        delta = prior.denoise(
            y[n] - args.tangent_eta * args.identity_sigma**2 * g[n],
            args.identity_sigma,
        ) - m
        raw_ratio[n] = local_tangent_normal_ratio(prior, m, g[n])
        cov_ratio[n] = local_tangent_normal_ratio(prior, m, cov_g)
        denoise_ratio[n] = local_tangent_normal_ratio(prior, m, delta)
    plot_tangent_filtering(raw_ratio, cov_ratio, figure_dir / "tangent_filtering.png")
    return {"raw_ratio": raw_ratio, "cov_ratio": cov_ratio, "denoise_ratio": denoise_ratio}


def write_report(out_path: Path, scale_results: dict, identity: dict, tangent: dict) -> None:
    lines = ["# One-step mechanism report", ""]
    for prior_name, data in scale_results["scale"].items():
        ratio = data["ratio"]
        lines.append(
            f"- {prior_name}: sigma_hat/sigma mean by d = "
            + ", ".join(f"{v:.3f}" for v in np.mean(ratio, axis=1))
        )
    lines.append(
        "- likelihood identity mean relative error by eta = "
        + ", ".join(f"{v:.3e}" for v in np.mean(identity["rel_errors"], axis=1))
    )
    lines.append(
        f"- tangent filtering median normal fraction: raw={np.median(tangent['raw_ratio']):.3f}, "
        f"Cg={np.median(tangent['cov_ratio']):.3f}, denoise-shift={np.median(tangent['denoise_ratio']):.3f}"
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="toy_blind_splitting/results")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--d-values", default="2,20,100")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-identity-samples", type=int, default=120)
    parser.add_argument("--curve-components", type=int, default=32)
    parser.add_argument("--grid-size", type=int, default=61)
    parser.add_argument("--sigma-min", type=float, default=0.01)
    parser.add_argument("--sigma-max", type=float, default=2.0)
    parser.add_argument("--true-sigma", type=float, default=0.35)
    parser.add_argument("--identity-d", type=int, default=40)
    parser.add_argument("--identity-sigma", type=float, default=0.25)
    parser.add_argument("--identity-etas", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 1e-1])
    parser.add_argument("--tangent-eta", type=float, default=1e-2)
    parser.add_argument("--priors", nargs="+", default=["subspace", "ellipse", "full"])
    args = parser.parse_args()

    if args.quick:
        args.d_values = "2,20"
        args.n_samples = min(args.n_samples, 64)
        args.n_identity_samples = min(args.n_identity_samples, 32)
        args.curve_components = min(args.curve_components, 12)
        args.grid_size = min(args.grid_size, 25)
        args.identity_d = min(args.identity_d, 16)

    out_dir = ensure_dir(args.out)
    figure_dir = ensure_dir(out_dir / "figures")
    data_dir = ensure_dir(out_dir / "data")

    scale = run_scale_concentration(args, figure_dir)
    identity = run_likelihood_identity(args, figure_dir)
    tangent = run_tangent_filter(args, figure_dir)

    np.savez_compressed(
        data_dir / "one_step_results.npz",
        d_values=scale["d_values"],
        subspace_ratio=scale["scale"].get("subspace", {}).get("ratio", np.empty((0, 0))),
        ellipse_ratio=scale["scale"].get("ellipse", {}).get("ratio", np.empty((0, 0))),
        full_ratio=scale["scale"].get("full", {}).get("ratio", np.empty((0, 0))),
        identity_etas=identity["etas"],
        identity_rel_errors=identity["rel_errors"],
        tangent_raw_ratio=tangent["raw_ratio"],
        tangent_cov_ratio=tangent["cov_ratio"],
        tangent_denoise_ratio=tangent["denoise_ratio"],
    )
    write_report(data_dir / "one_step_report.md", scale, identity, tangent)
    print(f"saved one-step results to {data_dir / 'one_step_results.npz'}")


if __name__ == "__main__":
    main()

