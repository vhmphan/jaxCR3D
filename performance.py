"""Generate one reproducible source and its noiseless diffusion cube."""
import argparse
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.source import make_source_prior
from src.solver import solve_u, sheet_source, residual_norm

def func_g_ana(x, y, z):
    L = x[-1]
    H = z[-1]

    g0 = 1.0
    q = jnp.pi / (jnp.sqrt(2.0) * L)

    # XY mode, including boundary points
    G = (
        jnp.cos(jnp.pi * x[:, None] / (2 * L))
        * jnp.cos(jnp.pi * y[None, :] / (2 * L))
    )

    # Exact continuum solution sampled on the grid
    Z = jnp.sinh(q * (H - jnp.abs(z))) / (2 * q * jnp.cosh(q * H))

    # Discrete thin-sheet source
    return g0 * G[1:-1, 1:-1], g0 * G[:, :, None] * Z[None, None, :]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=float, default=12.8)
    parser.add_argument("--H", type=float, default=6.4)
    args = parser.parse_args()
    L, H = args.L, args.H

    runtimes = []

    for dx in (0.4, 0.2, 0.1, 0.05):
        nx = ny = int(round(2*L/dx)) + 1
        nz = int(round(2*H/dx)) + 1
        x = jnp.linspace(-L, L, nx)
        y = jnp.linspace(-L, L, ny)
        z = jnp.linspace(-H, H, nz)

        g_exact, U_exact = func_g_ana(x, y, z)
        S = sheet_source(g_exact, nz, dx)

        # Warm up for this grid size.
        solve_u(S, dx).block_until_ready()

        # Measure runtime.
        start = time.perf_counter()
        U = solve_u(S, dx)
        U.block_until_ready()
        elapsed = time.perf_counter() - start

        size = np.log((nx-1)*(ny-1)*(nz-1)) / np.log(2.0)
        runtimes.append((size, elapsed))
        print(f"size={size:g}, shape={U.shape}, runtime={elapsed:.6f} s")

        # Save after each run so completed results are retained.
        np.savetxt(
            "runtimes.csv",
            runtimes,
            delimiter=" ",
            header="size,runtime_seconds",
            comments="",
            fmt=("%d", "%.8f"),
        )

    plt.plot(z, U[nx // 2, ny // 2, :], "o", label="Numerical")
    plt.plot(z, U_exact[nx // 2, ny // 2, :], "-", label="Exact")
    plt.xlim(-H, H)
    plt.ylim(0, 1.1 * np.max(U[nx // 2, ny // 2, :]))
    plt.xlabel("z")
    plt.ylabel("u(0, 0, z)")
    plt.legend()
    plt.savefig('fg_test_nCR_z.png')

    dU = np.abs(U[1:-1, 1:-1, nz//2] - U_exact[1:-1, 1:-1, nz//2]) / U_exact[1:-1, 1:-1, nz//2]

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)

    for ax, field, title, label in zip(
        axes,
        (U[1:-1, 1:-1, nz // 2], U_exact[1:-1, 1:-1, nz//2], np.log10(dU)),
        (f"U at z = {z[nz // 2]:g}", f"U_exact at z = {z[nz // 2]:g}", "Relative error"),
        ("U", "U_exact", "Relative error"),
    ):
        im = ax.pcolormesh(
            x[1:-1], y[1:-1], field.T,
            shading="auto", cmap="viridis",
            vmin=0,
        )
        # im = ax.pcolormesh(x[1:-1], y[1:-1], field.T, shading="auto", cmap="viridis")
        ax.set(xlabel="x", ylabel="y", title=title)
        ax.set_aspect("equal")
        cax = ax.inset_axes([1.05, 0, 0.05, 1])
        fig.colorbar(im, cax=cax, label=label)

    plt.savefig('fg_test_nCR_xy.png')


if __name__ == "__main__":
    main()




