"""Read mean HI/H2 cubes and plot their integrals along z.

Default output: gas_surface_density.png next to this script.
Use --mass for hydrogen mass surface density (excluding helium).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.constants import m_p
from astropy.io import fits
import jax
import jax.numpy as jnp
from jax.scipy.interpolate import RegularGridInterpolator


kpc_to_cm = 3.086e21 


DEFAULT_FITS = Path("/hpcwork/nx136056/Gas/mean_std_densities_xyz_global.fits")


@jax.jit
def interpolate_cube(cube, x, y, z, x_int, y_int, z_int):
    """Interpolate an (x, y, z) cube onto new coordinate arrays."""
    interpolator = RegularGridInterpolator(
        (x, y, z),
        cube,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    X, Y, Z = jnp.meshgrid(x_int, y_int, z_int, indexing="ij")
    points = jnp.stack((X, Y, Z), axis=-1)

    return interpolator(points)


def coordinates(header, axis):
    """Pixel-center coordinates of a one-based FITS axis, in kpc."""
    pixels = np.arange(header[f"NAXIS{axis}"]) + 1
    values = header[f"CRVAL{axis}"] + (pixels - header[f"CRPIX{axis}"]) * header[f"CDELT{axis}"]
    return (values * u.Unit(header[f"CUNIT{axis}"])).to_value(u.kpc)


def plot_surface(mean_hi, mean_h2, x, y, z, plot_name):

    dz = z[1] - z[0] # kpc

    surface_hi = np.sum(mean_hi, axis=2) * dz * kpc_to_cm
    surface_h2 = np.sum(mean_h2, axis=2) * dz * kpc_to_cm

    # H2 molecules contain two hydrogen atoms. No helium correction.
    factor = (m_p / u.cm**2).to_value(u.M_sun / u.pc**2)
    print(factor)
    surface_hi *= factor
    surface_h2 *= 2 * factor
    label = r"Surface density [M$_\odot$ pc$^{-2}$]"


    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5), constrained_layout=True
    )

    for ax, surface, title in zip(
        axes,
        (surface_hi, surface_h2),
        ("HI column density", "H₂ column density"),
    ):
        im = ax.imshow(
            surface.T,
            origin="lower",
            cmap="magma",
            interpolation="nearest",
            aspect="equal",
            vmin=0,
            vmax=20,
            extent=[x[0], x[-1], y[0], y[-1]]
        )
        ax.set(title=title, xlabel="x pixel", ylabel="y pixel")
        fig.colorbar(im, ax=ax, label=r"Column density [cm$^{-2}$]")

    fig.savefig("fg_gas_surface_%s.png" % plot_name, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", type=Path, default=DEFAULT_FITS)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("gas_surface_density.png"))
    args = parser.parse_args()

    # FITS axes (Z,Y,X) appear in NumPy as (X,Y,Z).
    with fits.open(args.fits, memmap=True) as hdus:
        hi = hdus["MEAN OF HI NUMBER DENSITY"]
        h2 = hdus["MEAN OF H2 NUMBER DENSITY"]
        mean_hi = hi.data
        mean_h2 = h2.data        

    mean_hi = jnp.asarray(np.asarray(mean_hi, dtype=np.float64))
    mean_h2 = jnp.asarray(np.asarray(mean_h2, dtype=np.float64))
    mean_hi = jnp.where(jnp.isnan(mean_hi), 0.0, mean_hi)
    mean_h2 = jnp.where(jnp.isnan(mean_h2), 0.0, mean_h2)
    mean_hi = mean_hi.transpose(1, 0, 2)
    mean_h2 = mean_h2.transpose(1, 0, 2)

    print(mean_hi.shape)
    
    x = coordinates(hi.header, 3) - 8.178
    y = coordinates(hi.header, 2)
    z = coordinates(hi.header, 1)

    L = 12.0
    H = 4.0
    dx_int = 0.1
    nx_int = ny_int = int(2.0*L/dx_int) + 1
    nz_int = int(2.0*H/dx_int) + 1
    x_int = jnp.linspace(-L, L, nx_int)
    y_int = jnp.linspace(-L, L, ny_int)
    z_int = jnp.linspace(-H, H, nz_int)

    mean_hi_int = interpolate_cube(mean_hi, x, y, z, x_int, y_int, z_int)
    mean_h2_int = interpolate_cube(mean_h2, x, y, z, x_int, y_int, z_int)

    mean_h_int = mean_hi_int + 2.0 * mean_h2_int 
    tag = np.format_float_positional(float(dx_int), trim="-").replace(".", "p")
    np.save(f"gas_SOD_dx_{tag}.npy", np.asarray(mean_h_int))#[0:2, 0:3, 0:4])

    plot_surface(mean_hi, mean_h2, x, y, z, 'origin')
    plot_surface(mean_hi_int, mean_h2_int, x_int, y_int, z_int, 'interp')

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5), constrained_layout=True
    )

    for ax, surface, title in zip(
        axes,
        (np.sum(mean_hi, axis=0), np.sum(mean_hi, axis=1)),
        ("HI column density", "H₂ column density"),
    ):
        im = ax.imshow(
            surface.T,
            origin="lower",
            cmap="magma",
            interpolation="nearest",
            aspect="equal",
            vmin=0,
            vmax=100,
            extent=[x[0], x[-1], z[0], z[-1]]
        )
        ax.set(title=title, xlabel="x pixel", ylabel="y pixel")
        fig.colorbar(im, ax=ax, label=r"Column density [cm$^{-2}$]")

    fig.savefig("fg_gas_surface_zprof.png", dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    main()
