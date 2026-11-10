"""Matrix-free Dirichlet diffusion, with implicit CG derivatives."""
from functools import partial

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg


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
    return S.at[1:-1, 1:-1, nz//2].set(g / dx)
