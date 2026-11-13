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
from src.solver import solve_u, solve_u_sg, sheet_source, residual_norm
from src.gsnr import func_g, func_gSNR_AMS09
from jax.scipy.special import gammaln

sigma_p = 30e-27 * (1.0 + 4.0**0.667*0.1) # cm^2
mp = 0.938272088                          # GeV/c^2

def func_Q0(E):
    """Injection spectrum; p in g cm/s, all constants in CGS."""
    epsilon = 0.1       # Acceleration efficiency for this species
    gamma = 4.23        # Injection spectral index
    Gamma_gamma = jnp.exp(gammaln((5.0 - gamma) / 2.0) + gammaln((gamma - 4.0) / 2.0) - 0.5 * jnp.log(jnp.pi)) / (2.0 * (gamma - 3.0))   # Γ(γ) 

    p = jnp.sqrt((E+mp)**2 - mp**2)   # GeV/c  
    beta = p / jnp.sqrt(p**2 + mp**2) # -> Particle's speed over speed of light
    E_SNR = 1e51 * 624.151            # GeV
    R_SNR = 0.03 * 1.0e6              # Myr^-1

    p = jnp.asarray(p)

    return (
        (epsilon * E_SNR * R_SNR / (Gamma_gamma * beta * mp**2)) * (p / mp)**(2.0-gamma) # GeV^-1 Myr^-1
    )


def func_D0(E):
    p = jnp.sqrt((E+mp)**2 - mp**2)

    """Diffusion coefficient in cm²/s; R in GV. Assumes protons."""
    D0 = 1.1e28/3.0198491e+29 # kpc²/Myr
    delta = 0.63              # δ
    Rb = 312.0                # GV
    delta_delta = 0.2         # Δδ
    s = 0.1                   # Break smoothness

    R = jnp.abs(jnp.asarray(p))

    # For protons, R in GV equals pc in GeV numerically.
    beta = R / jnp.sqrt(R**2 + mp**2)

    return (
        beta * D0 * R**delta
        / (1.0 + (R / Rb)**(delta_delta / s))**s # kpc^2/Myr
    )


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

    E = 10.0 # GeV
    beta = jnp.sqrt((E+mp)**2 - mp**2)/(E+mp)
    D0 = func_D0(E) # pc^2/yr
    Q0 = func_Q0(E) # GeV^-1

    S = sheet_source(g, nz, dx) * Q0 / D0
    source = make_source_prior((nx-2, ny-2), dx)
    latent = source.init(jax.random.PRNGKey(args.seed))
    nH = source(latent) * 1.0e3

    sg = sheet_source(nH * sigma_p * beta * 3.0e10, nz, dx) * 365.0 * 86400.0 * 1.0e6 / D0
    # U = solve_u(S, dx)
    U = solve_u_sg(S, sg, dx) / (3.086e21**3)

    print("%.2e Myr" % ((L)**2/D0))
    print("%.2e Myr" % (1.0 / (np.mean(nH) * sigma_p * 3.0e10 * 365.0 * 86400.0 * 1.0e6)))    
    print("Backend:", jax.default_backend())
    print("Cube shape:", U.shape)
    print("Relative stencil residual:", float(residual_norm(U, S, dx)))

    print("Mean of source surface density: ", np.mean(g))
    print("Mean of source surface density: ", np.mean(g))

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    tag = np.format_float_positional(dx, trim="-").replace(".", "p")
    np.save(output / f"prop_g_dx_{tag}.npy", np.asarray(g, dtype=np.float64))
    np.save(output / f"prop_ufull_dx_{tag}.npy", np.asarray(U, dtype=np.float64))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, xx, yy, field, title in zip(
        axes, (x[1:-1], x), (y[1:-1], y),
        (nH, U[:, :, nz//2]), ("Source g", f"U at z = {float(z[nz//2]):g}"),
    ):
        im = ax.pcolormesh(xx, yy, np.asarray(field).T, shading="auto")
        fig.colorbar(im, ax=ax)
        ax.set(xlabel="x", ylabel="y", title=title, aspect="equal")
    fig.savefig(output / f"fg_mock_dx_{tag}.png", dpi=150)
    plt.close(fig)
    print("Saved arrays and plot to", output)


if __name__ == "__main__":
    main()


