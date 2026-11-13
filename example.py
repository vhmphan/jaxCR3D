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
from src.gsnr import func_g, func_gSNR_AMS09

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dx", type=float, default=0.2)
    parser.add_argument("--L", type=float, default=12.0)
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

    # Distance squared from Earth.
    r2 = (
        (x[:, None, None] + 8.0)**2
        + y[None, :, None]**2
        + z[None, None, :]**2
    )

    mask = r2 <= 1.25**2

    # source = make_source_prior((nx-2, ny-2), dx)
    # latent = source.init(jax.random.PRNGKey(args.seed))
    # g = source(latent)

    # Compute the radial normalization.
    arr_rG = np.linspace(0.0, 17.2, 1000)
    phi = np.linspace(0.0, 2.0 * np.pi, 2000, endpoint=False)

    arr_frG = np.zeros_like(arr_rG)

    for i, radius in enumerate(arr_rG):
        arr_frG[i] = np.mean(func_g(radius, phi))

    X, Y = np.meshgrid(x, y, indexing="ij")  # Shape: (nx, ny)

    rG = np.hypot(X, Y)
    phiG = np.mod(np.arctan2(Y, X), 2.0 * np.pi)

    g = func_gSNR_AMS09(rG, phiG, arr_rG, arr_frG)[1:-1, 1:-1]
    g = jnp.rot90(g, k=1)
    print(g.shape)

    S = sheet_source(g, nz, dx)
    U = solve_u(S, dx)
    U_masked = np.where(mask, U, np.nan)
    print("Backend:", jax.default_backend())
    print("Cube shape:", U.shape)
    print("Relative stencil residual:", float(residual_norm(U, S, dx)))

    print("Mean of grouth truth: ", np.mean(g))

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    tag = np.format_float_positional(dx, trim="-").replace(".", "p")
    np.save(output / f"mock_g_dx_{tag}.npy", np.asarray(g, dtype=np.float64))
    np.save(output / f"mock_u_dx_{tag}.npy", np.asarray(U_masked, dtype=np.float64))
    np.save(output / f"mock_ufull_dx_{tag}.npy", np.asarray(U, dtype=np.float64))

    source = make_source_prior((nx-2, ny-2), dx)
    latent = source.init(jax.random.PRNGKey(args.seed))
    g = source(latent)

    print(np.mean(g))

    # fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    # for ax, xx, yy, field, title in zip(
    #     axes, (x[1:-1], x), (y[1:-1], y),
    #     (g, U[:, :, nz//2]), ("Source g", f"U at z = {float(z[nz//2]):g}"),
    # ):
    #     im = ax.pcolormesh(xx, yy, np.asarray(field).T, shading="auto")
    #     fig.colorbar(im, ax=ax)
    #     ax.set(xlabel="x", ylabel="y", title=title, aspect="equal")
    # fig.savefig(output / f"fg_mock_dx_{tag}.png", dpi=150)
    # plt.close(fig)
    # print("Saved arrays and plot to", output)


if __name__ == "__main__":
    main()


