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

path = Path("data/mock_u_dx_0p2.npy")

dx = float(path.stem.split("_dx_")[-1].replace("p", "."))
U_data = np.load(path)
U_mean = np.nanmean(U_data)
mask_data = np.isfinite(U_data)
nx, ny, nz = U_data.shape
L, H = (nx - 1) * dx / 2.0, (nz - 1) * dx / 2.0 
print(L, H)

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

    # def __call__(self, x):
    #     S = sheet_source(self.g(x), nz, dx)
    #     return solve_u(S, dx)
    def __call__(self, x):
        return self.masked(x)

forward = Forward(source)

# %%
key = random.PRNGKey(42)
key, k_f, k_n = random.split(key, 3)
synth_truth = U_data[mask_data]/U_mean 
synth_noise = 0.1 * synth_truth
synth_data = synth_truth + synth_noise

# Build likelihood
lh = jft.Gaussian(synth_data, noise_cov_inv=1.0 / synth_noise**2).amend(forward)

# Hyperparameters for nifty
n_vi_iterations = 5
delta = 1e-6
n_samples = 4

key, k_i, k_o = random.split(key, 3)
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
    sample_mode="linear_resample",
    odir="run",
    resume=False,
)

# Save samples into file
output = Path("run")
output.mkdir(parents=True, exist_ok=True)

g_samples = np.stack([
    np.asarray(forward.g(sample))
    for sample in samples
])

tag = np.format_float_positional(float(dx), trim="-").replace(".", "p")
np.save(output / f"g_samples_dx_{tag}.npy", g_samples * U_mean)
