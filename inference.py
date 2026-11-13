#!/usr/bin/env python3

# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

# %% [markdown]
# # 3D Tomography Example with known start- and end-points

# %%
import jax
import matplotlib.pyplot as plt
import nifty8.re as jft
import numpy as np
from jax import numpy as jnp
from jax import random
from pathlib import Path

from src.source import make_source_prior
from src.solver import solve_u, sheet_source, residual_norm

jax.config.update("jax_enable_x64", True)

def sampler(i):
    """30 % linear, 50 % nonlinear, 20 % update -- as a fraction, so it scales."""
    frac = i / n_vi_iterations
    if frac < 0.3:
        return "linear_resample"
    if frac < 0.8:
        return "nonlinear_resample"
    return "nonlinear_update"

path = Path("data/mock_u_dx_0p2.npy")

dx = float(path.stem.split("_dx_")[-1].replace("p", "."))
U_data = np.load(path)
mask_data = np.isfinite(U_data)
nx, ny, nz = U_data.shape
L, H = (nx - 1) * dx / 2.0, (nz - 1) * dx / 2.0 
print("Cosmic-ray halo:", L, "x", H, "kpc")

source = make_source_prior((nx-2, ny-2), dx)

class Forward(jft.Model):
    def __init__(self, g):
        self.g = g
        # Track a method with which a random input for the model. This is not
        # strictly required but is usually handy when building deep models.
        super().__init__(init=g.init)

    def CRcube(self, x):
        S = sheet_source(self.g(x), nz, dx)
        return solve_u(S, dx)

    def masked(self, x):
        U = self.CRcube(x)
        return U[mask_data]

    def __call__(self, x):
        return self.masked(x)

forward = Forward(source)

# %%
key = random.PRNGKey(42)
# key, k_f, k_n = random.split(key, 3)
# synth_truth = U_data[mask_data]/U_mean 
# synth_noise = 0.1 * synth_truth
# synth_data = synth_truth + synth_noise

synth_truth = jnp.asarray(U_data[mask_data])

relative_std = 0.1
sigma_log = jnp.sqrt(jnp.log1p(relative_std**2))

key, k_n = jax.random.split(key)
eps = jax.random.normal(
    k_n, synth_truth.shape, dtype=synth_truth.dtype
)

synth_data = synth_truth * jnp.exp(
    sigma_log * eps - 0.5 * sigma_log**2
)
synth_noise = synth_data - synth_truth

# Build likelihood
lh = jft.Gaussian(synth_data, noise_cov_inv=1.0 / synth_noise**2).amend(forward)

# Hyperparameters for nifty
n_vi_iterations = 3
delta = 1e-5
n_samples = 4

key, k_i, k_o = random.split(key, 3)

tag = np.format_float_positional(float(dx), trim="-").replace(".", "p")

def save_vi_samples(samples, state):
    output = Path("run") / f"vi-samples-{state.nit}"
    output.mkdir(parents=True, exist_ok=True)

    g_samples = np.stack([
        np.asarray(forward.g(sample))
        for sample in samples
    ])

    np.save(
        output / f"g_samples_dx_{tag}.npy",
        g_samples,
    )
    print(f"Saved VI iteration {state.nit} samples to {output}")

samples, state = jft.optimize_kl(
    lh,
    jft.Vector(lh.init(k_i)),
    n_total_iterations=n_vi_iterations,
    n_samples=lambda i: n_samples // 2 if i < 2 else n_samples,
    key=k_o,
    draw_linear_kwargs=dict(
        cg_name="SL",
        cg_kwargs=dict(absdelta=delta * jft.size(lh.domain) / 10.0, maxiter=100),
    ),
    nonlinearly_update_kwargs=dict(
        minimize_kwargs=dict(
            name="SN",
            xtol=delta,
            cg_kwargs=dict(name=None),
            maxiter=5,
        )
    ),
    kl_kwargs=dict(
        minimize_kwargs=dict(
            name="M", xtol=delta, cg_kwargs=dict(name="MCG"), maxiter=35
        )
    ),
    # sample_mode="linear_resample",
    sample_mode=sampler,
    odir="run",
    resume=False,
    callback=save_vi_samples,
)

# # Save samples into file
# output = Path("run")
# output.mkdir(parents=True, exist_ok=True)

# g_samples = np.stack([
#     np.asarray(forward.g(sample))
#     for sample in samples
# ])

# tag = np.format_float_positional(float(dx), trim="-").replace(".", "p")
# np.save(output / f"g_samples_dx_{tag}.npy", g_samples * U_mean)
