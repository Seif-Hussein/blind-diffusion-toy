"""Exact linear-Gaussian posterior and posterior-smoothed denoiser helpers."""

from __future__ import annotations

from posterior_bddm_oracle.src.torch_oracle_tools import exact_linear_posterior_gmm


def posterior_gmm_given_measurement(prior, A, y_obs, noise_std):
    posterior_prior, posterior = exact_linear_posterior_gmm(prior, A, y_obs, noise_std)
    return posterior_prior, posterior


def oracle_correction(prior, posterior_prior, Y, sigma):
    return posterior_prior.denoise(Y, sigma) - prior.denoise(Y, sigma)
