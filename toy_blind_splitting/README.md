# Closed-form toy for blind diffusion splitting

This package implements a neural-network-free simulation environment for testing
blind diffusion splitting with analytical Gaussian-mixture priors.

It includes:

- closed-form Gaussian-mixture noisy densities;
- non-blind Bayes denoisers `D_sigma(y)`;
- posterior covariance `C_sigma(y)`;
- MLE and Bayes blind denoisers over a sigma grid;
- exact linear-Gaussian inverse posteriors for diagnostics;
- scheduled, blind, oracle-scale, raw PnP, and covariance-filtered splitting baselines;
- one-step mechanism tests and closed-loop inverse-problem tests.

## Install

From the repository root:

```bash
python -m pip install -r toy_blind_splitting/requirements.txt
```

## Quick smoke tests

```bash
python -m toy_blind_splitting.src.experiments_one_step --quick
python -m toy_blind_splitting.src.experiments_closed_loop --quick
```

GPU/PyTorch high-dimensional smoke test:

```bash
python -m toy_blind_splitting.src.experiments_highd_likelihood_tilt_torch \
  --device auto \
  --d-values 50 \
  --eta-values 1e-3 \
  --n-trials 8 \
  --batch-size 8 \
  --n-steps 10
```

Outputs are written to:

```text
toy_blind_splitting/results/data/
toy_blind_splitting/results/figures/
```

The scripts save raw `.npz` files and short Markdown reports:

```text
toy_blind_splitting/results/data/one_step_results.npz
toy_blind_splitting/results/data/one_step_report.md
toy_blind_splitting/results/data/closed_loop_results.npz
toy_blind_splitting/results/data/closed_loop_report.md
```

## Larger examples

One-step BDDM and covariance-filter mechanism tests:

```bash
python -m toy_blind_splitting.src.experiments_one_step \
  --d-values 2,20,100,500 \
  --n-samples 400 \
  --n-identity-samples 200 \
  --curve-components 64 \
  --grid-size 81
```

Closed-loop ellipse experiment:

```bash
python -m toy_blind_splitting.src.experiments_closed_loop \
  --prior ellipse \
  --A-type random \
  --d-values 2,20,100 \
  --measurement-ratios 0.25,0.5,1.0 \
  --eta-values 0.001,0.003,0.01,0.03 \
  --beta-values 0.0,0.1,0.3 \
  --n-trials 50 \
  --n-steps 120 \
  --curve-components 64 \
  --grid-size 81
```

Full-dimensional negative control:

```bash
python -m toy_blind_splitting.src.experiments_closed_loop \
  --prior full \
  --d-values 20,100 \
  --measurement-ratios 0.5 \
  --eta-values 0.003,0.01,0.03 \
  --beta-values 0.0,0.1 \
  --n-trials 50
```

GPU-oriented eta-focused high-dimensional likelihood-tilt run:

```bash
python -m toy_blind_splitting.src.experiments_highd_likelihood_tilt_torch \
  --device auto \
  --d-values 500,1000 \
  --eta-values 1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2 \
  --methods scheduled_tilt,blind_mle_tilt,oracle_mle_tilt,raw_pnp_uncapped,raw_pnp_tuned,unscaled_noisy \
  --sigma-max 10.0 \
  --n-trials 32 \
  --batch-size 16 \
  --n-steps 80
```

The Colab notebook for the same GPU target is:

```text
toy_blind_splitting/notebooks/highd_likelihood_tilt_gpu_colab.ipynb
```

Public Colab URL:

```text
https://colab.research.google.com/github/Seif-Hussein/blind-diffusion-toy/blob/master/toy_blind_splitting/notebooks/highd_likelihood_tilt_gpu_colab.ipynb
```

This GPU path tests the lightweight first-order likelihood-tilt setting only.
It does not compute full posterior covariances or Kalman/finite-assimilation
updates.

## What to inspect

One-step results:

- `scale_hist_*.png`: histograms of `sigma_hat^2 / sigma^2`;
- `scale_mean_std_*.png`: mean and standard deviation of `sigma_hat / sigma`;
- `likelihood_tilt_identity.png`: first-order identity error versus `eta`;
- `tangent_filtering.png`: normal fraction of raw `g` versus `C_sigma(y) g`.

Closed-loop results:

- `scale_trajectory.png`: scheduled scale versus inferred active scale;
- `posterior_mean_mse_trajectory.png`: error to exact posterior mean;
- `measurement_mse_trajectory.png`: measurement consistency;
- `final_posterior_mean_mse.png`: final boxplot across trials;
- `latent_trajectories.png`: example latent trajectory on the ellipse prior.

## Notes

The implementation is intentionally dense and explicit. It uses eigendecomposed
component covariances to evaluate the closed-form Gaussian identities. Very large
combinations of `d`, mixture size, trial count, sigma-grid size, and closed-loop
steps can be expensive; start with the quick commands and scale up one axis at a
time.
