# Posterior-BDDM oracle diagnostics

This fork tests the viability question directly:

```text
Does a cheap splitting correction approximate the missing posterior-BDDM drift?
```

It reuses the closed-form Gaussian-mixture prior and exact linear-Gaussian
posterior code from `toy_blind_splitting`, but writes separate outputs under
`posterior_bddm_oracle/results/`.

## Quick smoke test

From the repository root:

```bash
python -m posterior_bddm_oracle.src.experiments_posterior_oracle --quick
```

This runs a small one-step correction diagnostic and a tiny closed-loop hierarchy.

## Main one-step oracle correction run

```bash
python -m posterior_bddm_oracle.src.experiments_posterior_oracle \
  --skip-closed-loop \
  --prior ellipse \
  --d-values 2,10,50,100,500 \
  --sigmas 0.03,0.1,0.3,1.0 \
  --eta-values 1e-4,3e-4,1e-3,3e-3,1e-2,3e-2 \
  --measurement-ratios 0.5 \
  --noise-stds 0.08 \
  --n-samples 500 \
  --split-methods gradient,pdhg,hqs \
  --pdhg-gammas 100 \
  --hqs-taus 1e-2
```

The key output is:

```text
posterior_bddm_oracle/results/data/one_step_posterior_oracle_report.md
```

Raw arrays are saved in `one_step_posterior_oracle.npz`.

## Reduced closed-loop hierarchy

```bash
python -m posterior_bddm_oracle.src.experiments_posterior_oracle \
  --skip-one-step \
  --prior ellipse \
  --closed-d-values 10,50 \
  --closed-measurement-ratios 0.5 \
  --closed-noise-stds 0.08 \
  --closed-eta-values 1e-3 \
  --closed-split gradient \
  --n-trials 20 \
  --n-steps 40
```

The hierarchy includes:

- `posterior_oracle`
- `prior_exact_cstar`
- `blind_prior_split`
- `blind_naive_force`
- `scheduled_split`
- `posterior_scale_split`
- `raw_pnp_hqs`

The key output is:

```text
posterior_bddm_oracle/results/data/closed_loop_posterior_oracle_report.md
```

Raw trajectories, scale diagnostics, correction alignment, posterior mean error,
posterior covariance error, component-weight KL, and measurement MSE are saved in
`closed_loop_posterior_oracle.npz`.

## CUDA / Colab port

The CUDA path ports the exact one-step posterior-correction oracle test to
PyTorch:

```bash
python -m posterior_bddm_oracle.src.experiments_posterior_oracle_torch --quick --device auto
```

On Colab, open:

```text
posterior_bddm_oracle/notebooks/posterior_bddm_cuda_colab.ipynb
```

For the calibrated closed-loop run only, open:

```text
posterior_bddm_oracle/notebooks/posterior_bddm_calibrated_closed_loop_colab.ipynb
```

The full Colab command is:

```bash
python -m posterior_bddm_oracle.src.experiments_posterior_oracle_torch \
  --device auto \
  --prior ellipse \
  --d-values 2,10,50,100,500 \
  --sigmas 0.03,0.1,0.3,1.0 \
  --eta-values 1e-4,3e-4,1e-3,3e-3,1e-2,3e-2 \
  --measurement-ratios 0.5 \
  --noise-stds 0.08 \
  --n-samples 500 \
  --split-methods gradient,pdhg,hqs
```

If the best eta is always the largest eta in the report, run an amplitude sweep:

```bash
python -m posterior_bddm_oracle.src.experiments_posterior_oracle_torch \
  --device auto \
  --prior ellipse \
  --d-values 50,100,500 \
  --sigmas 0.3,1.0 \
  --eta-values 0.01,0.03,0.1,0.3,1.0 \
  --measurement-ratios 0.5 \
  --noise-stds 0.08 \
  --n-samples 500 \
  --split-methods gradient,pdhg,hqs \
  --out posterior_bddm_oracle/results_cuda_eta_sweep
```

This runner computes exact dense-GMM posterior components, posterior-smoothed
denoisers, `c_star`, and split corrections using Torch tensors. It runs on CPU
locally and switches to CUDA automatically when a GPU runtime is available.

## CUDA closed-loop hierarchy

Before comparing split methods in closed loop, calibrate the posterior oracle:

```bash
python -m posterior_bddm_oracle.src.experiments_posterior_oracle_calibration_torch \
  --device auto \
  --prior ellipse \
  --d-values 100,500 \
  --h-values 0.005,0.01,0.02,0.05 \
  --beta-values 0,0.005,0.01,0.03,0.05 \
  --n-steps-values 40,80,160 \
  --measurement-ratios 0.5 \
  --noise-stds 0.08 \
  --n-trials 128 \
  --out posterior_bddm_oracle/results_cuda_oracle_calibration
```

Then run the reduced closed-loop posterior-BDDM hierarchy with calibrated
settings:

```bash
python -m posterior_bddm_oracle.src.experiments_posterior_closed_loop_torch \
  --device auto \
  --prior ellipse \
  --d-values 50,100,500 \
  --eta-values 0.03,0.1,0.3 \
  --measurement-ratios 0.5 \
  --noise-stds 0.08 \
  --n-trials 64 \
  --n-steps 40 \
  --init posterior \
  --beta 0.05 \
  --split gradient \
  --out posterior_bddm_oracle/results_cuda_closed_loop
```

This runner includes:

- `posterior_oracle`
- `prior_exact_cstar`
- `blind_split`
- `blind_naive_force`
- `scheduled_split`
- `posterior_scale_split`
- `raw_hqs`

The key output is:

```text
posterior_bddm_oracle/results_cuda_closed_loop/data/closed_loop_cuda_report.md
```

Use `--init posterior` for the oracle hierarchy because the posterior BDDM
drift assumes states lie near `pi_sigma`. The older `--init prior_mean` mode is
kept as a stress test, but it is not the clean oracle diagnostic.
