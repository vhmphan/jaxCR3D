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

def estimate_memory(func, *args):
    stats = func.lower(*args).compile().memory_analysis()

    if stats is None:
        return float("nan")  # Not supported by this backend.

    total = (
        stats.argument_size_in_bytes
        + stats.output_size_in_bytes
        + stats.temp_size_in_bytes
        - stats.alias_size_in_bytes
    )
    return total / 1024**3  # GiB

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

    performance = []

    # for dx in (0.2, 0.1, 0.05, 0.025):
    for dx in (0.1, 0.05):# (0.4, 0.2, 0.1, 0.05):
        nx = ny = int(round(2*L/dx)) + 1
        nz = int(round(2*H/dx)) + 1
        x = jnp.linspace(-L, L, nx)
        y = jnp.linspace(-L, L, ny)
        z = jnp.linspace(-H, H, nz)

        g_exact, U_exact = func_g_ana(x, y, z)
        S = sheet_source(g_exact, nz, dx)

        @jax.jit
        def forward(S):
            return solve_u(S, dx, tol=1.0e-3)

        # Solve: warm up, then measure.
        forward(S).block_until_ready()

        start = time.perf_counter()
        U = forward(S)
        U.block_until_ready()
        elapsed_solve = time.perf_counter() - start

        # JVP: solution + tangent for a source perturbation dS.
        dS = jnp.ones_like(S)

        @jax.jit
        def compute_jvp(S, dS):
            return jax.jvp(forward, (S,), (dS,))

        jax.block_until_ready(compute_jvp(S, dS))

        start = time.perf_counter()
        U_jvp, dU = compute_jvp(S, dS)
        jax.block_until_ready((U_jvp, dU))
        elapsed_jvp = time.perf_counter() - start

        # VJP: construct the pullback outside the timed section.
        U_vjp, pullback = jax.vjp(forward, S)
        U_vjp.block_until_ready()

        grad_U = jnp.ones_like(U_vjp)
        compute_vjp = jax.jit(pullback)

        jax.block_until_ready(compute_vjp(grad_U))

        start = time.perf_counter()
        grad_S, = compute_vjp(grad_U)
        grad_S.block_until_ready()
        elapsed_vjp = time.perf_counter() - start

        size = np.log2((nx-1)*(ny-1)*(nz-1))

        # Compiler estimates; keep these outside the runtime measurements.
        mem_solve = estimate_memory(forward, S)
        mem_jvp = estimate_memory(compute_jvp, S, dS)
        mem_vjp = estimate_memory(compute_vjp, grad_U)
       
        # memory_results.append((size, mem_solve, mem_jvp, mem_vjp))
        performance.append((size, elapsed_solve, elapsed_jvp, elapsed_vjp, mem_solve, mem_jvp, mem_vjp))

        print(
            f"size={size:g}, shape={U.shape}, "
            f"solve={elapsed_solve:.6f} s, "
            f"JVP={elapsed_jvp:.6f} s, "
            f"VJP={elapsed_vjp:.6f} s"
        )

        print(
            f"Estimated memory: "
            f"solve={mem_solve:.2f} GiB, "
            f"JVP={mem_jvp:.2f} GiB, "
            f"VJP={mem_vjp:.2f} GiB"
        )


        # Save after each grid size.
        np.savetxt(
            "performance.csv",
            performance,
            delimiter=" ",
            header="size solve_seconds jvp_seconds vjp_seconds solve_GiB jvp_GiB vjp_GiB",
            comments="",
            fmt=("%.4f", "%.8f", "%.8f", "%.8f", "%.4f", "%.4f", "%.4f"),
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
    print("\nMean relative error:", np.mean(dU))

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




