"""Generate one reproducible source and its noiseless diffusion cube."""
import argparse
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
    parser.add_argument("--dx", type=float, default=0.2)
    parser.add_argument("--L", type=float, default=10.0)
    parser.add_argument("--H", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    dx, L, H = args.dx, args.L, args.H
    nx = ny = int(round(2*L/dx)) + 1
    nz = int(round(2*H/dx)) + 1
    x = jnp.linspace(-L, L, nx)
    y = jnp.linspace(-L, L, ny)
    z = jnp.linspace(-H, H, nz)

    source = make_source_prior((nx-2, ny-2), dx)
    latent = source.init(jax.random.PRNGKey(args.seed))
    g = source(latent)
    S = sheet_source(g, nz, dx)
    U = solve_u(S, dx)
    print("Backend:", jax.default_backend())
    print("Cube shape:", U.shape)
    print("Relative stencil residual:", float(residual_norm(U, S, dx)))

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    tag = np.format_float_positional(dx, trim="-").replace(".", "p")
    np.save(output / f"mock_g_dx_{tag}.npy", np.asarray(g, dtype=np.float64))
    np.save(output / f"mock_u_dx_{tag}.npy", np.asarray(U, dtype=np.float64))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, xx, yy, field, title in zip(
        axes, (x[1:-1], x), (y[1:-1], y),
        (g, U[:, :, nz//2]), ("Source g", f"U at z = {float(z[nz//2]):g}"),
    ):
        im = ax.pcolormesh(xx, yy, np.asarray(field).T, shading="auto")
        fig.colorbar(im, ax=ax)
        ax.set(xlabel="x", ylabel="y", title=title, aspect="equal")
    fig.savefig(output / f"fg_mock_dx_{tag}.png", dpi=150)
    plt.close(fig)
    print("Saved arrays and plot to", output)

    # g_exact, U_exact = func_g_ana(x, y, z)
    # S = sheet_source(g_exact, nz, dx)
    # U = solve_u(S, dx)

    # plt.plot(z, U[nx // 2, ny // 2, :], "o", label="Numerical")
    # plt.plot(z, U_exact[nx // 2, ny // 2, :], "-", label="Exact")
    # plt.xlim(-H, H)
    # plt.ylim(0, 1.1 * np.max(U[nx // 2, ny // 2, :]))
    # plt.xlabel("z")
    # plt.ylabel("u(0, 0, z)")
    # plt.legend()
    # plt.savefig('fg_test_nCR_z.png')

    # dU = np.abs(U[1:-1, 1:-1, nz//2] - U_exact[1:-1, 1:-1, nz//2]) / U_exact[1:-1, 1:-1, nz//2]

    # fig, axes = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)

    # for ax, field, title, label in zip(
    #     axes,
    #     (U[1:-1, 1:-1, nz // 2], U_exact[1:-1, 1:-1, nz//2], np.log10(dU)),
    #     (f"U at z = {z[nz // 2]:g}", f"U_exact at z = {z[nz // 2]:g}", "Relative error"),
    #     ("U", "U_exact", "Relative error"),
    # ):
    #     im = ax.pcolormesh(
    #         x[1:-1], y[1:-1], field.T,
    #         shading="auto", cmap="viridis",
    #         vmin=0,
    #     )
    #     # im = ax.pcolormesh(x[1:-1], y[1:-1], field.T, shading="auto", cmap="viridis")
    #     ax.set(xlabel="x", ylabel="y", title=title)
    #     ax.set_aspect("equal")
    #     cax = ax.inset_axes([1.05, 0, 0.05, 1])
    #     fig.colorbar(im, cax=cax, label=label)

    # plt.savefig('fg_test_nCR_xy.png')


if __name__ == "__main__":
    main()


