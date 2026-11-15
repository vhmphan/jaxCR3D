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

import numpy as np
import matplotlib.pyplot as plt

def plot_mock_data(U_data, L, nx, dx, Lr=None, fg_name=None):
    tag = np.format_float_positional(float(dx), trim="-").replace(".", "p")
    x = y = np.linspace(-L, L, nx)

    U_full = np.load(f"data/mock_ufull_dx_{tag}.npy")
    U_data = np.asarray(U_data)

    iz = U_full.shape[2] // 2
    full_xy = U_full[:, :, iz]
    data_xy = U_data[:, :, iz]

    # Leave missing observations and zero-denominator points masked.
    relative_error = np.full(full_xy.shape, np.nan)
    valid = (
        np.isfinite(full_xy)
        & np.isfinite(data_xy)
        & (full_xy > 0)
    )
    relative_error[valid] = (
        np.abs(full_xy[valid] - data_xy[valid]) / full_xy[valid]
    )

    # Exact agreement is displayed at the bottom of the color scale.
    log_error = np.log10(np.maximum(relative_error, 1e-3))

    if Lr is None:
        xlim, ylim = (-L, L), (-L, L)
    else:
        xlim = (-8 - Lr, -8 + Lr)
        ylim = (-Lr, Lr)

    region = (
        (x[:, None] >= xlim[0]) & (x[:, None] <= xlim[1])
        & (y[None, :] >= ylim[0]) & (y[None, :] <= ylim[1])
    )

    values = np.concatenate([
        full_xy[region].ravel(),
        data_xy[region].ravel(),
    ])
    values = values[np.isfinite(values) & (values > 0)]
    vmin, vmax = np.percentile(values, [2, 98])

    fig, axes = plt.subplots(
        1, 3, figsize=(18, 5), constrained_layout=True
    )

    for i, (ax, field, title) in enumerate(zip(
        axes,
        (full_xy, data_xy, log_error),
        ("Full solution", "Mock data", "Log relative error"),
    )):
        im = ax.pcolormesh(
            x, y, field.T,
            shading="auto",
            cmap="magma",
            vmin=-3 if i == 2 else vmin,
            vmax=1 if i == 2 else vmax,
        )

        ax.set(
            xlabel="x (kpc)",
            ylabel="y (kpc)",
            title=title,
            aspect="equal",
            xlim=xlim,
            ylim=ylim,
        )

        if i == 0:
            im_U = im
        elif i == 2:
            im_error = im

    cbar = fig.colorbar(im_U, ax=axes[:2], pad=0.02)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()

    fig.colorbar(
        im_error, ax=axes[2], pad=0.02,
        ticks=[-3, -2, -1, 0, 1],
        label=r"$\log_{10}(|U_{\rm full}-U_{\rm data}|/U_{\rm full})$",
    )

    suffix = fg_name
    fig.savefig(f"data/fg_test_dx_{tag}{suffix}.png", dpi=150)
    plt.close(fig)

    return 0


def sampler(i):
    """30 % linear, 50 % nonlinear, 20 % update -- as a fraction, so it scales."""
    frac = i / n_vi_iterations
    if frac < 0.3:
        return "linear_resample"
    if frac < 0.8:
        return "nonlinear_resample"
    return "nonlinear_update"

path = Path("data/mock_u_dx_0p1.npy")

dx = float(path.stem.split("_dx_")[-1].replace("p", "."))
U_data = np.load(path)
mask_data = np.isfinite(U_data)
nx, ny, nz = U_data.shape
L, H = (nx - 1) * dx / 2.0, (nz - 1) * dx / 2.0 

print("Cosmic-ray halo:", L, "x", H, "kpc")
print("Cosmic-ray grid: (", nx, ny, nz,")")

source = make_source_prior((nx-2, ny-2), dx)

class Forward(jft.Model):
    def __init__(self, g):
        self.g = g

        # Track a method with which a random input for the model. 
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

# Build noise for mock data
key = random.PRNGKey(42)

# latent = source.init(jax.random.PRNGKey(43))
# g = source(latent)
# print(g[10, 10], g[-10, -10], np.mean(g), np.std(g))

# Data in log space.
truth = jnp.asarray(U_data)  # Must be positive.
relative_std = 0.1
sigma_log = jnp.sqrt(jnp.log1p(relative_std**2))

key, k_n = jax.random.split(key)
eps = jax.random.normal(k_n, truth.shape, dtype=truth.dtype)
synth_data = truth * jnp.exp(
    sigma_log * eps - 0.5 * sigma_log**2
)
log_data = jnp.log(synth_data) + 0.5 * sigma_log**2
log_data = log_data[mask_data]

plot_mock_data(U_data, L, nx, dx, Lr=None, fg_name="origin-full")
plot_mock_data(U_data, L, nx, dx, Lr=2, fg_name="origin-zoom")
plot_mock_data(synth_data, L, nx, dx, Lr=2, fg_name="noise-zoom")

# Predict log(U) at the observed locations.
log_forward = jft.Model(
    lambda latent: jnp.log(forward(latent)),
    domain=forward.domain,
    init=forward.init,
)

# Log-space noise variance is sigma_log**2.
lh = jft.Gaussian(
    log_data,
    noise_cov_inv=1.0 / (sigma_log*eps[mask_data])**2,
).amend(log_forward)

# Hyperparameters for nifty
n_vi_iterations = 20
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
