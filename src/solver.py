"""Matrix-free Dirichlet diffusion, with implicit CG derivatives."""
from functools import partial

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
from jax.scipy.sparse.linalg import bicgstab

def laplacian_operator(u, dx):
    """Discrete negative Laplacian on interior unknowns."""
    p = jnp.pad(u, 1)
    return (6*u - p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]
            - p[1:-1, 2:, 1:-1] - p[1:-1, :-2, 1:-1]
            - p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2]) / dx**2


@partial(jax.jit, static_argnames=("tol", "maxiter"))
def solve_u(S, dx, *, tol=1e-8, maxiter=2000):
    """Solve -laplacian(U)=S; ignore source values on boundary faces."""
    b = S[1:-1, 1:-1, 1:-1]
    u, _ = cg(lambda u: laplacian_operator(u, dx), b,
              tol=tol, maxiter=maxiter)
    return jnp.pad(u, 1)


def residual_norm(U, S, dx):
    """True interior residual: relative for nonzero RHS, absolute otherwise."""
    b = S[1:-1, 1:-1, 1:-1]
    r = laplacian_operator(U[1:-1, 1:-1, 1:-1], dx) - b
    norm_b = jnp.linalg.norm(b)
    return jnp.linalg.norm(r) / jnp.where(norm_b > 0, norm_b, 1.0)


def sheet_source(g, nz, dx):
    """Embed an interior XY source as g/dx on the central z plane."""
    S = jnp.zeros((g.shape[0]+2, g.shape[1]+2, nz), dtype=g.dtype)
    return S.at[1:-1, 1:-1, nz//2].set(g) # kpc^-2


@partial(jax.jit, static_argnames=("tol", "maxiter"))
def solve_u_sg(S, sg, dx, *, tol=1e-8, maxiter=2000):
    """Solve -laplacian(U) + sg*U = S with zero boundary values."""
    b = S[1:-1, 1:-1, 1:-1]
    sg_inner = sg[1:-1, 1:-1, 1:-1]

    def operator(u):
        return laplacian_operator(u, dx) + sg_inner * u

    u, _ = cg(operator, b, tol=tol, maxiter=maxiter)
    return jnp.pad(u, 1)


def diffusion_operator_ani(u, D, dx):
    """Discrete -div(D grad(u)) on interior unknowns.

    u: shape (nx-2, ny-2, nz-2).
    D: tuple (Dxx, Dyy, Dzz, Dxy, Dxz, Dyz),
       each with full shape (nx, ny, nz).
    """
    p = jnp.pad(u, 1)

    Dxx, Dyy, Dzz, Dxy, Dxz, Dyz = D
    tensor = (
        (Dxx, Dxy, Dxz),
        (Dxy, Dyy, Dyz),
        (Dxz, Dyz, Dzz),
    )

    def average_to_faces(a, axis):
        """Arithmetic average of adjacent grid values."""
        a = jnp.moveaxis(a, axis, 0)
        faces = 0.5 * (a[1:] + a[:-1])
        return jnp.moveaxis(faces, 0, axis)

    # Centred derivatives at grid points, used for transverse gradients.
    gradients = []
    for axis in range(3):
        a = jnp.moveaxis(p, axis, 0)
        g = (a[2:] - a[:-2]) / (2.0 * dx)
        g = jnp.pad(g, ((1, 1), (0, 0), (0, 0)))
        gradients.append(jnp.moveaxis(g, 0, axis))

    result = jnp.zeros_like(u)

    # Construct x-, y-, and z-face fluxes.
    for axis in range(3):
        flux = 0.0

        for direction in range(3):
            D_face = average_to_faces(tensor[axis][direction], axis)

            if direction == axis:
                # Normal derivative evaluated directly at the face.
                grad_face = jnp.diff(p, axis=axis) / dx
            else:
                # Transverse derivative interpolated to the same face.
                grad_face = average_to_faces(gradients[direction], axis)

            flux = flux + D_face * grad_face

        # Difference of neighbouring face fluxes.
        divergence = jnp.diff(flux, axis=axis) / dx

        # The difference already removes boundary entries along 'axis'.
        interior = [slice(1, -1)] * 3
        interior[axis] = slice(None)

        result = result - divergence[tuple(interior)]

    return result


@partial(jax.jit, static_argnames=("tol", "maxiter"))
def solve_u_ani(S, D, dx, *, tol=1e-8, maxiter=5000):
    """Return u with the same shape as S and zero boundary values."""
    b = S[1:-1, 1:-1, 1:-1]

    u, _ = bicgstab(
        lambda v: diffusion_operator_ani(v, D, dx),
        b,
        tol=tol,
        atol=0.0,
        maxiter=maxiter,
    )

    return jnp.pad(u, 1)

