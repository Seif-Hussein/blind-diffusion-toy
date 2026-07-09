"""Splitting algorithms for the closed-form blind diffusion toy."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .measurements import data_gradient
from .metrics import local_tangent_normal_ratio
from .priors import GaussianMixturePrior


def log_schedule(sigma_max: float, sigma_min: float, n_steps: int) -> np.ndarray:
    return np.geomspace(float(sigma_max), float(sigma_min), int(n_steps))


def initial_noisy_state(
    prior: GaussianMixturePrior,
    sigma0: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    return prior.mean + sigma0 * rng.normal(size=prior.d)


def _history() -> dict[str, list]:
    return {
        "x": [],
        "Y": [],
        "sigma_used": [],
        "sigma_hat": [],
        "grad_norm": [],
        "cov_grad_norm": [],
        "raw_normal_ratio": [],
        "cov_normal_ratio": [],
        "tr_C": [],
        "tr_C_plus": [],
        "kalman_update_norm": [],
        "assimilation_cond": [],
    }


def _append_history(
    hist: dict[str, list],
    x: np.ndarray,
    Y: Optional[np.ndarray],
    sigma_used: float,
    sigma_hat: float,
    g: Optional[np.ndarray] = None,
    cov_g: Optional[np.ndarray] = None,
    prior: Optional[GaussianMixturePrior] = None,
    tr_C: float = float("nan"),
    tr_C_plus: float = float("nan"),
    kalman_update_norm: float = float("nan"),
    assimilation_cond: float = float("nan"),
) -> None:
    hist["x"].append(np.asarray(x, dtype=float).copy())
    hist["Y"].append(None if Y is None else np.asarray(Y, dtype=float).copy())
    hist["sigma_used"].append(float(sigma_used))
    hist["sigma_hat"].append(float(sigma_hat))
    if g is None:
        hist["grad_norm"].append(float("nan"))
        hist["raw_normal_ratio"].append(float("nan"))
    else:
        hist["grad_norm"].append(float(np.linalg.norm(g)))
        hist["raw_normal_ratio"].append(
            float("nan") if prior is None else local_tangent_normal_ratio(prior, x, g)
        )
    if cov_g is None:
        hist["cov_grad_norm"].append(float("nan"))
        hist["cov_normal_ratio"].append(float("nan"))
    else:
        hist["cov_grad_norm"].append(float(np.linalg.norm(cov_g)))
        hist["cov_normal_ratio"].append(
            float("nan") if prior is None else local_tangent_normal_ratio(prior, x, cov_g)
        )
    hist["tr_C"].append(float(tr_C))
    hist["tr_C_plus"].append(float(tr_C_plus))
    hist["kalman_update_norm"].append(float(kalman_update_norm))
    hist["assimilation_cond"].append(float(assimilation_cond))


def _sigma_hat_safe(prior: GaussianMixturePrior, Y: Optional[np.ndarray]) -> float:
    if Y is None:
        return float("nan")
    return float(prior.sigma_mle(Y))


def _cov_action_safe(
    prior: GaussianMixturePrior,
    Y: Optional[np.ndarray],
    sigma: float,
    g: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if Y is None or g is None or not np.isfinite(sigma) or sigma <= 0.0:
        return None
    return prior.posterior_covariance_action(Y, float(sigma), g)


def scheduled_nonblind_forced_diffusion(
    prior: GaussianMixturePrior,
    A: np.ndarray,
    y_obs: np.ndarray,
    noise_std: float,
    schedule: np.ndarray,
    eta: float,
    h: float,
    beta: float,
    rng: Optional[np.random.Generator] = None,
    y0: Optional[np.ndarray] = None,
) -> dict:
    rng = np.random.default_rng() if rng is None else rng
    Y = initial_noisy_state(prior, float(schedule[0]), rng) if y0 is None else np.asarray(y0, dtype=float).copy()
    hist = _history()
    x0 = prior.denoise(Y, float(schedule[0]))
    _append_history(hist, x0, Y, float(schedule[0]), _sigma_hat_safe(prior, Y), prior=prior)

    for sigma in schedule:
        sigma = float(sigma)
        x = prior.denoise(Y, sigma)
        g = data_gradient(x, y_obs, A, noise_std)
        cov_g = _cov_action_safe(prior, Y, sigma, g)
        Y_tilde = Y - eta * sigma**2 * g
        x_tilde = prior.denoise(Y_tilde, sigma)
        noise_scale = np.sqrt(max(0.0, 2.0 * h * beta * sigma**2))
        Y = Y_tilde + h * (x_tilde - Y_tilde) + noise_scale * rng.normal(size=prior.d)
        x_next = prior.denoise(Y, sigma)
        _append_history(hist, x_next, Y, sigma, _sigma_hat_safe(prior, Y), g=g, cov_g=cov_g, prior=prior)

    x_final = prior.denoise(Y, float(schedule[-1]))
    return {"name": "scheduled_nonblind", "x_final": x_final, "Y_final": Y, "history": hist}


def _blind_eval(prior: GaussianMixturePrior, Y: np.ndarray, mode: str) -> tuple[np.ndarray, float]:
    if mode == "mle":
        sigma = float(prior.sigma_mle(Y))
        return prior.denoise(Y, sigma), sigma
    if mode == "bayes":
        post = prior.sigma_posterior(Y)
        sigma = float(post.mean)
        return prior.blind_denoise_bayes(Y), sigma
    raise ValueError(f"unknown blind mode: {mode}")


def blind_forced_diffusion(
    prior: GaussianMixturePrior,
    A: np.ndarray,
    y_obs: np.ndarray,
    noise_std: float,
    eta: float,
    h: float,
    beta: float,
    n_steps: int,
    sigma_min: float,
    rng: Optional[np.random.Generator] = None,
    y0: Optional[np.ndarray] = None,
    mode: str = "mle",
) -> dict:
    rng = np.random.default_rng() if rng is None else rng
    sigma0 = float(prior.sigma_grid[-1])
    Y = initial_noisy_state(prior, sigma0, rng) if y0 is None else np.asarray(y0, dtype=float).copy()
    hist = _history()
    x, sigma_hat = _blind_eval(prior, Y, mode)
    _append_history(hist, x, Y, sigma_hat, sigma_hat, prior=prior)

    for _ in range(int(n_steps)):
        x, sigma_hat = _blind_eval(prior, Y, mode)
        if sigma_hat <= sigma_min:
            break
        g = data_gradient(x, y_obs, A, noise_std)
        cov_g = _cov_action_safe(prior, Y, sigma_hat, g)
        Y_tilde = Y - eta * sigma_hat**2 * g
        x_tilde, sigma_tilde = _blind_eval(prior, Y_tilde, mode)
        noise_scale = np.sqrt(max(0.0, 2.0 * h * beta * sigma_tilde**2))
        Y = Y_tilde + h * (x_tilde - Y_tilde) + noise_scale * rng.normal(size=prior.d)
        x_next, sigma_next = _blind_eval(prior, Y, mode)
        _append_history(hist, x_next, Y, sigma_tilde, sigma_next, g=g, cov_g=cov_g, prior=prior)

    x_final, sigma_final = _blind_eval(prior, Y, mode)
    return {"name": f"blind_{mode}", "x_final": x_final, "Y_final": Y, "sigma_final": sigma_final, "history": hist}


def oracle_scale_nonblind_forced_diffusion(
    prior: GaussianMixturePrior,
    A: np.ndarray,
    y_obs: np.ndarray,
    noise_std: float,
    eta: float,
    h: float,
    beta: float,
    n_steps: int,
    sigma_min: float,
    rng: Optional[np.random.Generator] = None,
    y0: Optional[np.ndarray] = None,
) -> dict:
    # This uses the closed-form MLE scale but applies the non-blind D_sigma.
    return blind_forced_diffusion(
        prior,
        A,
        y_obs,
        noise_std,
        eta=eta,
        h=h,
        beta=beta,
        n_steps=n_steps,
        sigma_min=sigma_min,
        rng=rng,
        y0=y0,
        mode="mle",
    ) | {"name": "oracle_scale_nonblind"}


def raw_pnp_splitting(
    prior: GaussianMixturePrior,
    A: np.ndarray,
    y_obs: np.ndarray,
    noise_std: float,
    schedule: np.ndarray,
    eta: float,
    rng: Optional[np.random.Generator] = None,
    x0: Optional[np.ndarray] = None,
    y0_for_init: Optional[np.ndarray] = None,
    rho_factor: float = 0.0,
) -> dict:
    rng = np.random.default_rng() if rng is None else rng
    if x0 is None:
        if y0_for_init is not None:
            x = prior.denoise(y0_for_init, float(schedule[0]))
        else:
            x = prior.mean.copy()
    else:
        x = np.asarray(x0, dtype=float).copy()
    hist = _history()
    _append_history(hist, x, None, float(schedule[0]), float("nan"), prior=prior)

    for sigma in schedule:
        sigma = float(sigma)
        g = data_gradient(x, y_obs, A, noise_std)
        q = x - eta * g
        denoiser_input = q + rho_factor * sigma * rng.normal(size=prior.d)
        cov_g = _cov_action_safe(prior, denoiser_input, sigma, g)
        x = prior.denoise(denoiser_input, sigma)
        _append_history(hist, x, denoiser_input, sigma, _sigma_hat_safe(prior, denoiser_input), g=g, cov_g=cov_g, prior=prior)
    return {"name": f"raw_pnp_rho_{rho_factor:g}", "x_final": x, "Y_final": None, "history": hist}


def blind_clean_pnp_splitting(
    prior: GaussianMixturePrior,
    A: np.ndarray,
    y_obs: np.ndarray,
    noise_std: float,
    n_steps: int,
    eta: float,
    rng: Optional[np.random.Generator] = None,
    x0: Optional[np.ndarray] = None,
    y0_for_init: Optional[np.ndarray] = None,
    mode: str = "mle",
) -> dict:
    """Clean-space data step followed by blind denoising."""

    _ = np.random.default_rng() if rng is None else rng
    if x0 is None:
        if y0_for_init is not None:
            x, sigma_hat = _blind_eval(prior, y0_for_init, mode)
        else:
            x = prior.mean.copy()
            sigma_hat = float(prior.sigma_mle(x))
    else:
        x = np.asarray(x0, dtype=float).copy()
        sigma_hat = float(prior.sigma_mle(x))
    hist = _history()
    _append_history(hist, x, x, sigma_hat, sigma_hat, prior=prior)

    for _step in range(int(n_steps)):
        x, sigma_hat = _blind_eval(prior, x, mode)
        g = data_gradient(x, y_obs, A, noise_std)
        cov_g = _cov_action_safe(prior, x, sigma_hat, g)
        q = x - eta * g
        x, sigma_next = _blind_eval(prior, q, mode)
        _append_history(hist, x, q, sigma_hat, sigma_next, g=g, cov_g=cov_g, prior=prior)
    x_final, sigma_final = _blind_eval(prior, x, mode)
    return {
        "name": f"blind_clean_pnp_{mode}",
        "x_final": x_final,
        "Y_final": x,
        "sigma_final": sigma_final,
        "history": hist,
    }


def raw_noisy_shift_without_sigma2(
    prior: GaussianMixturePrior,
    A: np.ndarray,
    y_obs: np.ndarray,
    noise_std: float,
    schedule: np.ndarray,
    eta: float,
    h: float,
    beta: float,
    rng: Optional[np.random.Generator] = None,
    y0: Optional[np.ndarray] = None,
) -> dict:
    """Noisy-coordinate data shift without the ``sigma^2`` likelihood-tilt scale."""

    rng = np.random.default_rng() if rng is None else rng
    Y = initial_noisy_state(prior, float(schedule[0]), rng) if y0 is None else np.asarray(y0, dtype=float).copy()
    hist = _history()
    x0 = prior.denoise(Y, float(schedule[0]))
    _append_history(hist, x0, Y, float(schedule[0]), _sigma_hat_safe(prior, Y), prior=prior)

    for sigma in schedule:
        sigma = float(sigma)
        x = prior.denoise(Y, sigma)
        g = data_gradient(x, y_obs, A, noise_std)
        cov_g = _cov_action_safe(prior, Y, sigma, g)
        Y_tilde = Y - eta * g
        x_tilde = prior.denoise(Y_tilde, sigma)
        noise_scale = np.sqrt(max(0.0, 2.0 * h * beta * sigma**2))
        Y = Y_tilde + h * (x_tilde - Y_tilde) + noise_scale * rng.normal(size=prior.d)
        x_next = prior.denoise(Y, sigma)
        _append_history(hist, x_next, Y, sigma, _sigma_hat_safe(prior, Y), g=g, cov_g=cov_g, prior=prior)

    x_final = prior.denoise(Y, float(schedule[-1]))
    return {"name": "raw_noisy_no_sigma2", "x_final": x_final, "Y_final": Y, "history": hist}


def covariance_filtered_clean_update(
    prior: GaussianMixturePrior,
    A: np.ndarray,
    y_obs: np.ndarray,
    noise_std: float,
    schedule: np.ndarray,
    eta: float,
    rng: Optional[np.random.Generator] = None,
    y0: Optional[np.ndarray] = None,
    sigma_mode: str = "schedule",
    log_full_covariance: bool = False,
) -> dict:
    rng = np.random.default_rng() if rng is None else rng
    Y = initial_noisy_state(prior, float(schedule[0]), rng) if y0 is None else np.asarray(y0, dtype=float).copy()
    hist = _history()
    x = prior.denoise(Y, float(schedule[0]))
    _append_history(hist, x, Y, float(schedule[0]), _sigma_hat_safe(prior, Y), prior=prior)

    for sigma_sched in schedule:
        if sigma_mode == "mle":
            sigma = float(prior.sigma_mle(Y))
        elif sigma_mode == "schedule":
            sigma = float(sigma_sched)
        else:
            raise ValueError(f"unknown sigma_mode: {sigma_mode}")
        x = prior.denoise(Y, sigma)
        g = data_gradient(x, y_obs, A, noise_std)
        tr_C = float("nan")
        if log_full_covariance:
            C = prior.posterior_covariance(Y, sigma)
            cov_g = C @ g
            tr_C = float(np.trace(C))
        else:
            cov_g = prior.posterior_covariance_action(Y, sigma, g)
        x = x - eta * cov_g
        Y = x.copy()
        _append_history(hist, x, Y, sigma, _sigma_hat_safe(prior, Y), g=g, cov_g=cov_g, prior=prior, tr_C=tr_C)
    return {"name": f"covariance_filtered_{sigma_mode}", "x_final": x, "Y_final": Y, "history": hist}


def finite_kalman_assimilation(
    prior: GaussianMixturePrior,
    A: np.ndarray,
    y_obs: np.ndarray,
    noise_std: float,
    schedule: np.ndarray,
    alpha: float,
    h: float,
    beta: float,
    rng: Optional[np.random.Generator] = None,
    y0: Optional[np.ndarray] = None,
    mode: str = "mle",
    lift: str = "residual",
) -> dict:
    """Finite tilted-posterior/Kalman data assimilation in the BDDM state.

    ``mode="mle"`` uses the MLE active scale and then calls the non-blind
    analytical denoiser/covariance at that scale. ``mode="schedule"`` uses the
    supplied schedule. The covariance update is used for diagnostics and for
    the optional covariance-shrink re-noising lift.
    """

    rng = np.random.default_rng() if rng is None else rng
    Y = initial_noisy_state(prior, float(schedule[0]), rng) if y0 is None else np.asarray(y0, dtype=float).copy()
    hist = _history()

    def eval_state(state: np.ndarray, sigma_hint: float) -> tuple[np.ndarray, float]:
        if mode == "mle":
            sigma = float(prior.sigma_mle(state))
        elif mode == "schedule":
            sigma = float(sigma_hint)
        else:
            raise ValueError(f"unknown finite assimilation mode: {mode}")
        return prior.denoise(state, sigma), sigma

    x0, sigma0 = eval_state(Y, float(schedule[0]))
    _append_history(hist, x0, Y, sigma0, _sigma_hat_safe(prior, Y), prior=prior)

    R_scale = (noise_std**2) / float(alpha)
    eye_m = np.eye(A.shape[0])

    for sigma_hint in schedule:
        x, sigma = eval_state(Y, float(sigma_hint))
        C = prior.posterior_covariance(Y, sigma)
        tr_C = float(np.trace(C))
        residual = A @ x - y_obs
        g = data_gradient(x, y_obs, A, noise_std)
        cov_g = C @ g

        CAt = C @ A.T
        S = A @ CAt + R_scale * eye_m
        S = 0.5 * (S + S.T)
        try:
            solved_residual = np.linalg.solve(S, residual)
            solved_AC = np.linalg.solve(S, A @ C)
            cond = float(np.linalg.cond(S))
        except np.linalg.LinAlgError:
            solved_residual = np.linalg.pinv(S) @ residual
            solved_AC = np.linalg.pinv(S) @ (A @ C)
            cond = float("inf")

        kalman_update = CAt @ solved_residual
        x_plus = x - kalman_update
        C_plus = C - CAt @ solved_AC
        C_plus = 0.5 * (C_plus + C_plus.T)
        tr_C_plus = float(max(np.trace(C_plus), 0.0))

        if lift == "residual":
            Y_plus = x_plus + (Y - x)
        elif lift == "fresh":
            Y_plus = x_plus + sigma * rng.normal(size=prior.d)
        elif lift == "cov_shrink":
            shrink = np.sqrt(np.clip(tr_C_plus / max(tr_C, 1e-15), 0.0, 1.0))
            Y_plus = x_plus + sigma * shrink * rng.normal(size=prior.d)
        else:
            raise ValueError(f"unknown lift: {lift}")

        x_lift, sigma_lift = eval_state(Y_plus, float(sigma_hint))
        noise_scale = np.sqrt(max(0.0, 2.0 * h * beta * sigma_lift**2))
        Y = Y_plus + h * (x_lift - Y_plus) + noise_scale * rng.normal(size=prior.d)
        _append_history(
            hist,
            x_lift,
            Y,
            sigma_lift,
            sigma_lift if mode == "mle" else _sigma_hat_safe(prior, Y),
            g=g,
            cov_g=cov_g,
            prior=prior,
            tr_C=tr_C,
            tr_C_plus=tr_C_plus,
            kalman_update_norm=float(np.linalg.norm(kalman_update)),
            assimilation_cond=cond,
        )

    x_final, sigma_final = eval_state(Y, float(schedule[-1]))
    return {
        "name": f"finite_kalman_{mode}_alpha_{alpha:g}_{lift}",
        "x_final": x_final,
        "Y_final": Y,
        "sigma_final": sigma_final,
        "history": hist,
    }
