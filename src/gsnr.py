import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt


# Constants
pi = np.pi
rG0 = 3.0                  # kpc: radius of the Galactic bar
phiG0 = np.deg2rad(70.0)    # radians
pbar = 12.6                 # pitch angle in degrees
kbar = np.tan(np.deg2rad(pbar))
fwhm = 1.177


# Gaussian profile
def func_gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2)


# Surface density of one spiral arm
# i = 1, 2, 3, or 4; rG in kpc; phiG in radians
def func_spiral(i, rG, phiG):
    spiral = 0.0
    phi_arm = phiG0 + (i - 1) * pi / 2.0

    for j in range(3):
        phi_min = phi_arm - j * 2.0 * pi
        phi_max = phi_arm + (2 - j) * 2.0 * pi

        inside = (phiG > phi_min) & (phiG < phi_max)

        arm_radius = rG0 * np.exp(
            kbar * (phiG + j * 2.0 * pi - phi_arm)
        )

        profile = func_gaussian(rG, arm_radius, 0.5 * fwhm)
        spiral = spiral + np.where(inside, profile, 0.0)

    return spiral


# Auxiliary functions for the Galactic bar
def func_blurA(x, x1, x2, w):
    return 0.5 * (erf((x - x1) / w) + erf((x2 - x) / w))


def func_blurB(x, mu, sigma):
    return 0.5 * (
        erf((x - mu + sigma) / 0.1)
        + erf((mu + sigma - x) / 0.1)
    )


# Surface density of the Galactic bar
def func_galbar(rG, phiG):
    blurA = func_blurA(rG, -1.0, 3.5, 0.5 * fwhm)
    blurB = func_blurB(
        rG * np.sin(phiG - phiG0),
        0.0,
        0.5 * fwhm,
    )

    return blurA * blurB


# SNR surface density from Case & Bhattacharya (1998)
def func_gSNR_CAB98(rG):
    A = 1.96
    rG0_SNR = 17.2
    theta0 = 0.08
    B = 0.13

    gSNR = A * np.sin(pi * rG / rG0_SNR + theta0) * np.exp(-B * rG)

    gSNR = np.where((rG >= 16.8) | (gSNR < 0.0), 0.0, gSNR)

    return gSNR / 335.42271571658637  # kpc^-2


# Spiral arms + Galactic bar, before radial correction
def func_g(rG, phiG):
    g = 0.0

    for i in range(1, 5):
        g = g + func_spiral(i, rG, phiG)

    g = g + func_galbar(rG, phiG)

    return np.minimum(g, 1.0)


# Linear interpolation in radius
def func_interp_1D_lin(arr_rG, arr_fiG, rG):
    return np.interp(rG, arr_rG, arr_fiG)


# Corrected SNR surface density
def func_gSNR_AMS09(rG, phiG, arr_rG, arr_frG):
    radial_average = func_interp_1D_lin(arr_rG, arr_frG, rG)
    target_density = func_gSNR_CAB98(rG)

    # Divide only where the denominator is positive.
    correction = np.zeros_like(radial_average, dtype=float)
    np.divide(
        target_density,
        radial_average,
        out=correction,
        where=radial_average > 0.0,
    )

    gSNR = func_g(rG, phiG) * correction

    valid = (rG < 17.2) & (gSNR >= 0.0) & np.isfinite(gSNR)

    return np.where(valid, gSNR, 0.0)  # kpc^-2

# # Compute the radial normalization.
# arr_rG = np.linspace(0.0, 17.2, 1000)
# phi = np.linspace(0.0, 2.0 * np.pi, 2000, endpoint=False)

# arr_frG = np.zeros_like(arr_rG)

# for i, radius in enumerate(arr_rG):
#     arr_frG[i] = np.mean(func_g(radius, phi))

# # Grid matching your source-file convention.
# nx, ny = 257, 257
# x = np.linspace(-10.0, 10.0, nx)
# y = np.linspace(-10.0, 10.0, ny)

# X, Y = np.meshgrid(x, y, indexing="ij")  # Shape: (nx, ny)

# rG = np.hypot(X, Y)
# phiG = np.mod(np.arctan2(Y, X), 2.0 * np.pi)

# g = func_gSNR_AMS09(rG, phiG, arr_rG, arr_frG)

# # Add the sample axis: (nx, ny) -> (1, nx, ny).
# source = g[np.newaxis, :, :]

# np.savez_compressed(
#     "sources_spiral.npz",
#     x=x,
#     y=y,
#     source=source,
# )

# print("Saved source shape:", source.shape)

# # Plot.
# fig, ax = plt.subplots(figsize=(7, 6))

# image = ax.pcolormesh(
#     x, y, g.T,
#     shading="auto",
#     cmap="magma",
# )

# ax.plot(0.0, 8.0, "yx", markersize=8, label="Sun")
# ax.set_xlabel("x [kpc]")
# ax.set_ylabel("y [kpc]")
# ax.set_aspect("equal")
# ax.legend()

# fig.colorbar(image, ax=ax, label=r"$g_{\mathrm{SNR}}$ [kpc$^{-2}$]")
# fig.tight_layout()
# fig.savefig("gSNR_AMS09.png", dpi=200)
# # plt.show()