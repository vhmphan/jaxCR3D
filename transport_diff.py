import jax.numpy as jnp
from src.gmfmod import gmf_cartesian
import numpy as np
import matplotlib.pyplot as plt
from src.solver import sheet_source, diffusion_operator_ani, solve_u_ani
from src.spiral import func_g, func_gSNR_AMS09
from mpl_toolkits.axes_grid1 import make_axes_locatable


import math
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def toroidal_test_grid(
    L, H, dx, *,
    r_in=1.0, r_out=4.0,
    a=1.0, Dpar=1.0, Dperp=0.01, U0=1.0,
):
    """Return x, y, z, b, D, u_exact, S_exact, inside.

    x, y, z: 1D coordinate vectors including box boundaries.
    b: (bx, by, bz), each shape (nx, ny, nz).
    D: (Dxx, Dyy, Dzz, Dxy, Dxz, Dyz), same shapes.
    u_exact, S_exact: full arrays, zero outside the open annular cylinder.
    inside: boolean mask of the open annular cylinder.

    b and D are extended outside the domain to keep arrays finite.
    At r=0, b=(1,0,0) is an arbitrary convention inside the excluded hole.
    Requires positive Dpar and Dperp. Grid creation uses Python scalars.
    """
    if min(L, H, dx) <= 0:
        raise ValueError("L, H, and dx must be positive.")
    if not (0 < r_in < r_out <= L):
        raise ValueError("Require 0 < r_in < r_out <= L.")
    if Dpar <= 0 or Dperp <= 0:
        raise ValueError("Require Dpar > 0 and Dperp > 0.")
    if not (0 <= a <= 1) or U0 < 0:
        raise ValueError("Require 0 <= a <= 1 and U0 >= 0.")

    def coordinate(half_width):
        intervals = round(2.0 * half_width / dx)
        if intervals < 2 or not math.isclose(
            intervals * dx, 2.0 * half_width,
            rel_tol=1e-10, abs_tol=1e-12,
        ):
            raise ValueError("2*L and 2*H must be integer multiples of dx.")
        return jnp.linspace(-half_width, half_width, intervals + 1)

    x, y, z = coordinate(L), coordinate(L), coordinate(H)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    r = jnp.sqrt(X**2 + Y**2)
    r_safe = jnp.where(r > 0.0, r, 1.0)
    inside = (r > r_in) & (r < r_out) & (jnp.abs(Z) < H)

    # Toroidal field direction b = e_phi; the axis is outside the domain.
    bx = jnp.where(r > 0.0, -Y / r_safe, 1.0)
    by = X / r_safe
    bz = jnp.zeros_like(X)
    b = (bx, by, bz)

    delta = Dpar - Dperp
    D = (
        Dperp + delta * bx**2,
        Dperp + delta * by**2,
        Dperp + delta * bz**2,
        delta * bx * by,
        delta * bx * bz,
        delta * by * bz,
    )

    alpha = jnp.pi / jnp.log(r_out / r_in)
    k = jnp.pi / (2.0 * H)
    epsilon = a * Dperp * alpha**2 / (Dperp * alpha**2 + Dpar)

    cos_phi = X / r_safe
    f = jnp.sin(alpha * jnp.log(r_safe / r_in))
    base = U0 * f * jnp.cos(k * Z)

    u_exact = base * (1.0 + epsilon * cos_phi)
    S_exact = Dperp * base * (
        alpha**2 / r_safe**2 * (1.0 + a * cos_phi)
        + k**2 * (1.0 + epsilon * cos_phi)
    )

    u_exact = jnp.where(inside, u_exact, 0.0)
    S_exact = jnp.where(inside, S_exact, 0.0)
    return b, D, u_exact, S_exact, inside


def helical_test_grid(
    L, H, dx, *,
    a=1.0,
    Dpar=1.0,
    Dperp=0.01,
):
    """Manufactured steady diffusion problem with helical field lines.

    Returns:
        x, y, z: coordinate vectors
        b: (bx, by, bz), each shape (nx, ny, nz)
        D: (Dxx, Dyy, Dzz, Dxy, Dxz, Dyz), same shapes
        u_exact, S: arrays of shape (nx, ny, nz)

    All lengths must use the same units.
    Dpar and Dperp are spatially constant.
    """
    if min(L, H, dx, a) <= 0:
        raise ValueError("L, H, dx, and a must be positive.")
    if Dpar <= 0 or Dperp < 0:
        raise ValueError("Require Dpar > 0 and Dperp >= 0.")

    # Include both boundaries while retaining exactly the requested dx.
    def coordinate(half_width):
        intervals = round(2.0 * half_width / dx)
        if intervals < 2 or not math.isclose(
            intervals * dx, 2.0 * half_width,
            rel_tol=1e-10, abs_tol=1e-12,
        ):
            raise ValueError("2*L and 2*H must be integer multiples of dx.")
        return jnp.linspace(-half_width, half_width, intervals + 1)

    x = coordinate(L)
    y = coordinate(L)
    z = coordinate(H)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

    # Helical unit vector: b = v / |v|.
    norm2 = 1.0 + (X**2 + Y**2) / a**2
    norm = jnp.sqrt(norm2)

    bx = -Y / (a * norm)
    by =  X / (a * norm)
    bz = 1.0 / norm
    b = (bx, by, bz)

    # Diffusion tensor.
    delta = Dpar - Dperp
    D = (
        Dperp + delta * bx**2,
        Dperp + delta * by**2,
        Dperp + delta * bz**2,
        delta * bx * by,
        delta * bx * bz,
        delta * by * bz,
    )

    # Exact solution and analytic derivatives.
    kx = ky = jnp.pi / (2.0 * L)
    kz = jnp.pi / (2.0 * H)

    cx, cy, cz = jnp.cos(kx * X), jnp.cos(ky * Y), jnp.cos(kz * Z)
    sx, sy, sz = jnp.sin(kx * X), jnp.sin(ky * Y), jnp.sin(kz * Z)

    u_exact = cx * cy * cz

    ux = -kx * sx * cy * cz
    uy = -ky * cx * sy * cz

    uxx = -kx**2 * u_exact
    uyy = -ky**2 * u_exact
    uzz = -kz**2 * u_exact

    uxy = kx * ky * sx * sy * cz
    uxz = kx * kz * sx * cy * sz
    uyz = ky * kz * cx * sy * sz

    # (v . grad)^2 u, where v = (-y/a, x/a, 1).
    # The terms -X*ux - Y*uy account for the varying field direction.
    directional_second = (
        (Y**2 * uxx + X**2 * uyy
         - 2.0 * X * Y * uxy - X * ux - Y * uy) / a**2
        + 2.0 * (-Y * uxz + X * uyz) / a
        + uzz
    )

    # S = -div(D grad(u_exact)).
    S = (
        -Dperp * (uxx + uyy + uzz)
        - delta * directional_second / norm2
    )

    return b, D, u_exact, S

# Domain size in kpc
L = 10.0
H = 4.0
dx = 0.1

# Number of grid points
nx = ny = int(2.0*L/dx) + 1
nz = int(2.0*H/dx) + 1

x = jnp.linspace(-L, L, nx)
y = jnp.linspace(-L, L, ny)
z = jnp.linspace(-H, H, nz)

# Each coordinate array has shape (Nx, Ny, Nz).
X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

# Positions: shape (Nx, Ny, Nz, 3)
xyz = jnp.stack((X, Y, Z), axis=-1)

# Magnetic field in microgauss: shape (Nx, Ny, Nz, 3)
B = gmf_cartesian(xyz)

# Three cubes, each with shape (Nx, Ny, Nz)
Bx = B[..., 0]
By = B[..., 1]
Bz = B[..., 2]

iz = Bx.shape[2] // 2

slices = [
    np.asarray(Bx[:, :, iz]),
    np.asarray(By[:, :, iz]),
    np.asarray(Bz[:, :, iz]),
]

# Shared symmetric colour scale for all three components
vmax = max(np.max(np.abs(s)) for s in slices)

fig, axes = plt.subplots(
    1, 3, figsize=(15, 4), sharex=True, sharey=True,
    constrained_layout=True,
)

for ax, data, label in zip(axes, slices, [r"$B_x$", r"$B_y$", r"$B_z$"]):
    im = ax.imshow(
        data.T,
        origin="lower",
        extent=[-L, L, -L, L],
        cmap="RdBu_r",
        vmin=-vmax/4.0,
        vmax=vmax/4.0,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_title(label)
    ax.set_xlabel(r"$x\;[\mathrm{kpc}]$")

axes[0].set_ylabel(r"$y\;[\mathrm{kpc}]$")
fig.colorbar(im, ax=axes, label=r"$B\;[\mu\mathrm{G}]$", shrink=0.85)

plt.savefig("fg_gmf.png")

# Extract the slices before forming the products.
Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)
bx = Bx/Bmag
by = By/Bmag
bz = Bz/Bmag

print(np.mean(bx**2))
print(np.mean(by**2))
print(np.mean(bz**2))
print(np.mean(np.abs(bx*by)))
print(np.mean(np.abs(bx*bz)))
print(np.mean(np.abs(by*bz)))

bx_mid = np.asarray(bx[:, :, iz])
by_mid = np.asarray(by[:, :, iz])
bz_mid = np.asarray(bz[:, :, iz])

components = [
    (bx_mid**2,      r"$D_{xx}$"),
    (by_mid**2,      r"$D_{yy}$"),
    (bz_mid**2,      r"$D_{zz}$"),
    (bx_mid*by_mid,  r"$D_{xy}$"),
    (bx_mid*bz_mid,  r"$D_{xz}$"),
    (by_mid*bz_mid,  r"$D_{yz}$"),
]

# Shared symmetric colour scale
vmax = max(np.max(np.abs(data)) for data, _ in components)
vmax = vmax if vmax > 0 else 1.0

fig, axes = plt.subplots(
    2, 3, figsize=(14, 8),
    sharex=True, sharey=True,
    constrained_layout=True,
)

for ax, (data, title) in zip(axes.flat, components):
    im = ax.imshow(
        data.T,
        origin="lower",
        extent=[-L, L, -L, L],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_title(title)

for ax in axes[-1, :]:
    ax.set_xlabel(r"$x\;[\mathrm{kpc}]$")

for ax in axes[:, 0]:
    ax.set_ylabel(r"$y\;[\mathrm{kpc}]$")

fig.colorbar(
    im, ax=axes.ravel().tolist(),
    label=r"$D_{ij}=b_i b_j$", shrink=0.85,
)

plt.savefig("fg_Dij.png")



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
g = jnp.ones_like(g)

dx = x[1] - x[0]
S = sheet_source(g, nz, dx)

Dxx = bx**2
Dyy = by**2
Dzz = bz**2
Dxy = bx*by
Dxz = bx*bz
Dyz = by*bz

D = (Dxx, Dyy, Dzz, Dxy, Dxz, Dyz)

u = solve_u_ani(S, D, dx)

b = S[1:-1, 1:-1, 1:-1]
Au = diffusion_operator_ani(u[1:-1, 1:-1, 1:-1], D, dx)

relative_residual = (
    jnp.linalg.norm(Au - b)
    / jnp.maximum(jnp.linalg.norm(b), 1e-30)
)
print("Relative residual:", relative_residual)

# u = solve_u_ani(S, Dxx, Dyy, Dzz, Dxy, Dxz, Dyz, dx, tol=1.0e-8, maxiter=2000)

# b = S[1:-1, 1:-1, 1:-1]
# Au = diffusion_operator_ani(
#     u[1:-1, 1:-1, 1:-1],
#     Dxx, Dyy, Dzz, Dxy, Dxz, Dyz, dx,
# )

# relative_residual = (
#     jnp.linalg.norm(Au - b)
#     / jnp.maximum(jnp.linalg.norm(b), 1e-30)
# )
# print("Relative residual:", relative_residual)

slices = [
    u[:, :, nz // 2],  # XY
    u[:, ny // 2, :],  # XZ
    u[nx // 2, :, :],  # YZ
]

extents = [
    [-L, L, -L, L],
    [-L, L, -H, H],
    [-L, L, -H, H],
]

titles = ["XY midplane", "XZ midplane", "YZ midplane"]
labels = [("x", "y"), ("x", "z"), ("y", "z")]

# Shared colour scale
vmin = min(s.min() for s in slices)
vmax = max(s.max() for s in slices)

# Match row heights to the physical aspect ratios.
height_ratios = [
    abs((extent[3] - extent[2]) / (extent[1] - extent[0]))
    for extent in extents
]

fig, axes = plt.subplots(
    3, 1,
    figsize=(8, 12),
    constrained_layout=True,
    gridspec_kw={"height_ratios": height_ratios},
)

for ax, data, extent, title, (xlabel, ylabel) in zip(
    axes, slices, extents, titles, labels
):
    im = ax.imshow(
        data.T,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
    )

    ax.set_title(title)
    ax.set_xlabel(f"{xlabel} [kpc]")
    ax.set_ylabel(f"{ylabel} [kpc]")

    # Fixed colorbar width and gap, both in inches.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.18, pad=0.12)
    fig.colorbar(im, cax=cax, label=r"$u$")

fig.savefig("fg_u_ani.png", dpi=300, bbox_inches="tight")

# slices = [
#     u[:, :, nz // 2],  # XY
#     u[:, ny // 2, :],  # XZ
#     u[nx // 2, :, :],  # YZ
# ]

# extents = [
#     [-L, L, -L, L],
#     [-L, L, -H, H],
#     [-L, L, -H, H],
# ]

# titles = ["XY midplane", "XZ midplane", "YZ midplane"]
# labels = [("x", "y"), ("x", "z"), ("y", "z")]

# # Shared colour scale
# vmin = min(s.min() for s in slices)
# vmax = max(s.max() for s in slices)

# # Match row heights to the physical aspect ratios so all panels have
# # the same width while retaining equal spatial scales on both axes.
# height_ratios = [
#     abs((extent[3] - extent[2]) / (extent[1] - extent[0]))
#     for extent in extents
# ]
# fig, axes = plt.subplots(
#     3, 1, figsize=(8, 12), constrained_layout=True,
#     gridspec_kw={"height_ratios": height_ratios},
# )

# for ax, data, extent, title, (xlabel, ylabel) in zip(
#     axes, slices, extents, titles, labels
# ):
#     im = ax.imshow(
#         data.T,
#         origin="lower",
#         extent=extent,
#         cmap="viridis",
#         vmin=vmin,
#         vmax=vmax,
#         aspect="equal",
#         interpolation="nearest",
#     )
#     ax.set_title(title)
#     ax.set_xlabel(f"{xlabel} [kpc]")
#     ax.set_ylabel(f"{ylabel} [kpc]")

#     ax.set_title(title)
#     ax.set_xlabel(f"{xlabel} [kpc]")
#     ax.set_ylabel(f"{ylabel} [kpc]")
#     fig.colorbar(im, ax=ax, label=r"$u$", shrink=0.85)

# plt.savefig("fg_u_ani.png")

#fig.colorbar(im, ax=axes.tolist(), label=r"$u$", shrink=0.85)
# plt.savefig("fg_u_ani.png")



# b, D, u_exact, S = helical_test_grid(
#     L, H, dx,
#     a=1.0,
#     Dpar=1.0,
#     Dperp=0.01,
# )

b, D, u_exact, S_exact, inside = toroidal_test_grid(
    L=L, H=H, dx=dx,
    r_in=1.0, r_out=4.0,
    a=1.0, Dpar=1.0, Dperp=0.01,
)

bx, by, bz = b
u = solve_u_ani(S, D, dx, tol=1e-10, maxiter=10000)

interior = (slice(1, -1),) * 3

relative_error = (
    jnp.linalg.norm(u[interior] - u_exact[interior])
    / jnp.linalg.norm(u_exact[interior])
)

Au = diffusion_operator_ani(u[interior], D, dx)
relative_residual = (
    jnp.linalg.norm(Au - S[interior])
    / jnp.linalg.norm(S[interior])
)

print("Grid shape:", u.shape)
print("Relative solution error:", relative_error)
print("Relative residual:", relative_residual)


U = np.asarray(u[:, :, iz])
U_exact = np.asarray(u_exact[:, :, iz])

# Mask points where the exact solution is effectively zero.
threshold = 1e-12 * np.max(np.abs(U_exact))
relative_error = np.full_like(U_exact, np.nan, dtype=float)
np.divide(
    np.abs(U - U_exact),
    np.abs(U_exact),
    out=relative_error,
    where=np.abs(U_exact) > threshold,
)

# Shared colour range for the first two panels.
vmin = min(U.min(), U_exact.min())
vmax = max(U.max(), U_exact.max())

fig, axes = plt.subplots(
    1, 3, figsize=(18, 5), constrained_layout=True
)

panels = [
    (U, r"$u$", "viridis", vmin, vmax),
    (U_exact, r"$u_{\mathrm{exact}}$", "viridis", vmin, vmax),
    (
        relative_error,
        r"$|u-u_{\mathrm{exact}}|/|u_{\mathrm{exact}}|$",
        "magma", 0.0, None,
    ),
]

for ax, (data, title, cmap, lower, upper) in zip(axes, panels):
    im = ax.imshow(
        data.T,
        origin="lower",
        extent=[-L, L, -L, L],
        aspect="equal",
        interpolation="nearest",
        cmap=cmap,
        vmin=lower,
        vmax=upper,
    )
    ax.set_title(title)
    ax.set_xlabel(r"$x\;[\mathrm{kpc}]$")
    ax.set_ylabel(r"$y\;[\mathrm{kpc}]$")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.18, pad=0.10)
    fig.colorbar(im, cax=cax)

plt.savefig("fg_u_ani_comp.png")