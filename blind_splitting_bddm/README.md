# Posterior-guided BDDM splitting diagnostics

This is a Colab-first CUDA/PyTorch environment for testing whether splitting
updates can approximate the missing posterior-smoothed BDDM correction

```text
c_star(Y, sigma; y) = E[X | Y, y] - E[X | Y].
```

The first stage uses only closed-form Gaussian-mixture priors and exact
linear-Gaussian posterior oracles. It does not train neural networks.

## Colab

Open:

```text
https://colab.research.google.com/github/Seif-Hussein/blind-diffusion-toy/blob/agent/colab-cuda-oracle/blind_splitting_bddm/notebooks/posterior_guided_bddm_colab.ipynb
```

The notebook clones this repo automatically into:

```text
/content/blind_splitting_bddm
```

## Install Locally

From the repository root:

```bash
python -m pip install -r blind_splitting_bddm/requirements.txt
```

## Smoke Suite

```bash
python -m blind_splitting_bddm.src.runners.run_smoke \
  --config blind_splitting_bddm/configs/smoke.yaml \
  --device cuda \
  --save_dir blind_splitting_bddm/results/smoke
```

This runs:

- BDDM prior scale inference;
- exact posterior correction oracle;
- PDHG dual-law diagnostics;
- reduced closed-loop hierarchy;
- combined Markdown report.

## Individual Runners

Prior scale inference:

```bash
python -m blind_splitting_bddm.src.runners.run_prior_scale_tests \
  --config blind_splitting_bddm/configs/gmm_prior_scale.yaml \
  --device cuda
```

Posterior correction oracle:

```bash
python -m blind_splitting_bddm.src.runners.run_posterior_correction_tests \
  --config blind_splitting_bddm/configs/posterior_correction_oracle.yaml \
  --device cuda
```

PDHG dual-law analysis:

```bash
python -m blind_splitting_bddm.src.runners.run_dual_law_tests \
  --config blind_splitting_bddm/configs/dual_law_pdhg_quadratic.yaml \
  --device cuda
```

Closed-loop hierarchy:

```bash
python -m blind_splitting_bddm.src.runners.run_closed_loop \
  --config blind_splitting_bddm/configs/closed_loop_reduced.yaml \
  --device cuda
```

Aggregate reports:

```bash
python -m blind_splitting_bddm.src.runners.make_report \
  --config blind_splitting_bddm/configs/smoke.yaml \
  --save_dir blind_splitting_bddm/results/smoke
```

## Output

Each runner writes:

```text
results/.../data/
results/.../figures/
results/.../reports/
```

Checkpoint-like partial files are written by the expensive posterior-correction
and closed-loop runners after each condition. The dual-law and prior-scale
runners save complete `.npz` results and Markdown reports.

## What To Read

The report separates:

- scale mismatch: `sigma_hat`, entropy, boundary-hit rates;
- optimization-induced bias: split perturbation bias ratio;
- anisotropic perturbation: covariance anisotropy ratio;
- dual-memory effects: PDHG identity residual and `delta_k`;
- posterior correction error: `||c_split - c_star|| / ||c_star||` and cosine.

The approach is promising only if scale inference is well behaved, PDHG dual
tracking is numerically verified, the split correction aligns with `c_star`, and
the closed-loop hierarchy moves toward the posterior oracle without relying on
boundary-saturated sigma estimates.
