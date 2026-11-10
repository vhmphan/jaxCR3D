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


if __name__ == "__main__":
    main()


