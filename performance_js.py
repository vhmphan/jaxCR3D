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

import sys
sys.path.insert(0, "/home/nx136056/jsolver")

from jsolver.grid_1d import grid_1D as grid_1Djs
from jsolver.solver_3d import solve_3d as solve_3djs

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

    performancejs = []

    for dx in (0.4, 0.2, 0.1, 0.05):
        nx = ny = int(round(2*L/dx)) + 1
        nz = int(round(2*H/dx)) + 1
        x = jnp.linspace(-L, L, nx)
        y = jnp.linspace(-L, L, ny)
        z = jnp.linspace(-H, H, nz)

        g_exact, U_exact = func_g_ana(x, y, z)
        S = sheet_source(g_exact, nz, dx)

        grid_xjs = grid_1Djs(nx - 1, xb=-L, xe=L)
        grid_yjs = grid_1Djs(ny - 1, xb=-L, xe=L)
        grid_zjs = grid_1Djs(nz - 1, xb=-H, xe=H)

        # Fixed iterations are required for reverse-mode differentiation.
        cyclesjs = 20

        @jax.jit
        def forwardjs(Sjs):
            # Convert (x,y,z) to (z,y,x), add ghosts, and reverse RHS sign.
            rhsjs = -jnp.pad(Sjs.transpose(2, 1, 0), 1)

            phijs, countjs = solve_3djs(
                grid_xjs, grid_yjs, grid_zjs,
                fixed=True,
                max_iters_outer=cyclesjs,
                max_iters_inner=1,
                phi=jnp.zeros_like(rhsjs),
                rhs=rhsjs,
                lam=0.0,
                D_xx=1.0,
                D_yy=1.0,
                D_zz=1.0,
            )

            # Remove ghosts and restore (x,y,z).
            return phijs[1:-1, 1:-1, 1:-1].transpose(2, 1, 0)

        # Solve: compile and warm up before timing.
        Sjs = S
        forwardjs(Sjs).block_until_ready()

        startjs = time.perf_counter()
        Ujs = forwardjs(Sjs)
        Ujs.block_until_ready()
        elapsed_solvejs = time.perf_counter() - startjs

        # JVP.
        dSjs = jnp.ones_like(Sjs)

        @jax.jit
        def compute_jvpjs(Sjs, dSjs):
            return jax.jvp(forwardjs, (Sjs,), (dSjs,))

        jax.block_until_ready(compute_jvpjs(Sjs, dSjs))

        startjs = time.perf_counter()
        U_jvpjs, dUjs = compute_jvpjs(Sjs, dSjs)
        jax.block_until_ready((U_jvpjs, dUjs))
        elapsed_jvpjs = time.perf_counter() - startjs

        # VJP: construct the pullback outside the timed section.
        U_vjpjs, pullbackjs = jax.vjp(forwardjs, Sjs)
        U_vjpjs.block_until_ready()
        grad_Ujs = jnp.ones_like(U_vjpjs)
        compute_vjpjs = jax.jit(pullbackjs)

        jax.block_until_ready(compute_vjpjs(grad_Ujs))

        startjs = time.perf_counter()
        grad_Sjs, = compute_vjpjs(grad_Ujs)
        grad_Sjs.block_until_ready()
        elapsed_vjpjs = time.perf_counter() - startjs

        # Compiler memory estimates, using your existing helper.
        mem_solvejs = estimate_memory(forwardjs, Sjs)
        mem_jvpjs = estimate_memory(compute_jvpjs, Sjs, dSjs)
        mem_vjpjs = estimate_memory(compute_vjpjs, grad_Ujs)

        # Check accuracy independently of the fixed iteration count.
        residualjs = (
            6 * Ujs[1:-1, 1:-1, 1:-1]
            - Ujs[:-2, 1:-1, 1:-1]
            - Ujs[2:, 1:-1, 1:-1]
            - Ujs[1:-1, :-2, 1:-1]
            - Ujs[1:-1, 2:, 1:-1]
            - Ujs[1:-1, 1:-1, :-2]
            - Ujs[1:-1, 1:-1, 2:]
        ) / dx**2 - Sjs[1:-1, 1:-1, 1:-1]

        relative_residualjs = (
            jnp.linalg.norm(residualjs)
            / jnp.linalg.norm(Sjs[1:-1, 1:-1, 1:-1])
        )

        sizejs = np.log2((nx - 1) * (ny - 1) * (nz - 1))
        performancejs.append((
            sizejs,
            elapsed_solvejs, elapsed_jvpjs, elapsed_vjpjs,
            mem_solvejs, mem_jvpjs, mem_vjpjs,
        ))

        print(
            f"jsolver: shape={Ujs.shape}, "
            f"solve={elapsed_solvejs:.6f} s, "
            f"JVP={elapsed_jvpjs:.6f} s, "
            f"VJP={elapsed_vjpjs:.6f} s, "
            f"relative residual={float(relative_residualjs):.3e}"
        )
        print(
            f"jsolver estimated memory: "
            f"solve={mem_solvejs:.2f} GiB, "
            f"JVP={mem_jvpjs:.2f} GiB, "
            f"VJP={mem_vjpjs:.2f} GiB"
        )

        np.savetxt(
            "performance_js.csv",
            performancejs,
            header=(
                "size solve_seconds jvp_seconds vjp_seconds "
                "solve_GiB jvp_GiB vjp_GiB"
            ),
            comments="",
            fmt="%.8f",
        )

    plt.plot(z, Ujs[nx // 2, ny // 2, :], "o", label="Numerical")
    plt.plot(z, U_exact[nx // 2, ny // 2, :], "-", label="Exact")
    plt.xlim(-H, H)
    plt.ylim(0, 1.1 * np.max(Ujs[nx // 2, ny // 2, :]))
    plt.xlabel("z")
    plt.ylabel("u(0, 0, z)")
    plt.legend()
    plt.savefig('fg_test_nCR_z_js.png')

    dU = np.abs(Ujs[1:-1, 1:-1, nz//2] - U_exact[1:-1, 1:-1, nz//2]) / U_exact[1:-1, 1:-1, nz//2]
    print("\nMean relative error:", np.mean(dU))

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)

    for ax, field, title, label in zip(
        axes,
        (Ujs[1:-1, 1:-1, nz // 2], U_exact[1:-1, 1:-1, nz//2], np.log10(dU)),
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

    plt.savefig('fg_test_nCR_xy_js.png')


if __name__ == "__main__":
    main()




