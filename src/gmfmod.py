"""Simplified JF12 field from AL-Zetoun and Achterberg MNRAS 2021."""

from typing import NamedTuple
import jax
import jax.numpy as jnp


class GMFParameters(NamedTuple):
    h: float = 0.4
    r0: float = 5.0
    pitch: float = 11.5 * jnp.pi / 180.0
    bx0: float = 4.6
    hx: float = 2.9
    rx: float = 4.8
    tan_i0: float = 1.15  # Use the screenshot's rounded value literally.
    arm_strengths: tuple = (0.1, 3.0, -0.9, -0.8, -2.0, -4.2, 0.0, 2.7)
    arm_boundaries: tuple = (5.1, 6.3, 7.1, 8.3, 9.8, 11.4, 12.7, 15.5)


DEFAULT = GMFParameters()


@jax.jit
def spiral_arm_index(r, phi, params=DEFAULT):
    """Zero-based arm index; boundaries cross the negative x axis.

    Boundaries follow r(phi)=r_boundary*exp((phi-pi)*tan(p)).
    At an exact boundary select the arm on its larger-radius side.
    Requires positive radius and positive pitch; defaults use JF12 values.
    """
    r, phi = jnp.broadcast_arrays(jnp.asarray(r), jnp.asarray(phi))
    safe_r = jnp.where(r > 0, r, 1.0)
    edges = jnp.asarray(params.arm_boundaries)
    period = 2.0 * jnp.pi * jnp.tan(params.pitch)
    log_crossing = jnp.log(safe_r) - (phi - jnp.pi) * jnp.tan(params.pitch)
    # Choose the outermost crossing below the last boundary. Full-turn
    # wrapping makes this periodic in phi, including the atan2 branch cut.
    turns = jnp.floor((log_crossing - jnp.log(edges[-1])) / period) + 1.0
    crossing = jnp.exp(log_crossing - turns * period)
    return jnp.clip(jnp.searchsorted(edges, crossing, side="right"), 0, 7)


def _x_geometry(r, height, params):
    """Footpoint, sin(i), cos(i), inner-region flag and rp/r limit."""
    transition_r = params.rx + height / params.tan_i0
    inner = r < transition_r
    ratio_inner = params.rx / transition_r
    rp = jnp.where(inner, r * ratio_inner, r - height / params.tan_i0)
    # tan(i)=rx*tan(i0)/rp in the inner region; avoid dividing by rp.
    numerator = params.rx * params.tan_i0
    denominator = jnp.sqrt(numerator**2 + rp**2)
    outer_norm = jnp.sqrt(1.0 + params.tan_i0**2)
    sin_i = jnp.where(inner, numerator / denominator, params.tan_i0 / outer_norm)
    cos_i = jnp.where(inner, rp / denominator, 1.0 / outer_norm)
    return rp, sin_i, cos_i, inner, ratio_inner


@jax.jit
def gmf_cylindrical(r, phi, z, params=DEFAULT, b0_disk=None):
    """Return (...,3) components (Br,Bphi,Bz), in microgauss.

    r, phi, z broadcast together. If b0_disk is supplied, it must broadcast
    to the coordinate shape and replaces the automatic arm strength lookup.
    This permits a chosen spiral section or a user-defined arm map.
    Physical parameters h, r0, bx0, hx, rx and tan_i0 are positive.
    """
    r, phi, z = jnp.broadcast_arrays(jnp.asarray(r), jnp.asarray(phi), jnp.asarray(z))
    in_disk = jnp.abs(z) <= params.h
    safe_r = jnp.where(r > 0, r, 1.0)
    if b0_disk is None:
        b0_disk = jnp.asarray(params.arm_strengths)[spiral_arm_index(r, phi, params)]
    bd = jnp.where(in_disk & (r > 0), jnp.asarray(b0_disk) * params.r0 / safe_r, 0.0)

    # Inside: evaluate the footpoint at h, independent of the local z.
    # Outside: evaluate it at |z|, with the original JF12 flux dilution.
    height = jnp.maximum(jnp.abs(z), params.h)
    rp, sin_i, cos_i, inner, ratio_inner = _x_geometry(r, height, params)
    bx_base = params.bx0 * jnp.exp(-rp / params.hx)
    dilution = jnp.where(inner, ratio_inner**2, rp / safe_r)
    bx = bx_base * jnp.where(in_disk, 1.0, dilution)
    br_x = jnp.where(in_disk, 0.0, jnp.sign(z) * bx * cos_i)

    br = bd * jnp.sin(params.pitch) + br_x
    bphi = bd * jnp.cos(params.pitch)
    bz = bx * sin_i
    return jnp.stack((br, bphi, bz), axis=-1)


@jax.jit
def gmf_cartesian(xyz, params=DEFAULT, b0_disk=None):
    """Evaluate positions of shape (...,3); return the same shape, in uG.

    Any grid/batch dimensions are supported directly, without vmap.
    Spatial derivatives on the cylindrical axis are not defined here.
    """
    xyz = jnp.asarray(xyz)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    r = jnp.hypot(x, y)
    phi = jnp.arctan2(y, x)
    b = gmf_cylindrical(r, phi, z, params, b0_disk)
    br, bphi, bz = b[..., 0], b[..., 1], b[..., 2]
    c, s = jnp.cos(phi), jnp.sin(phi)
    return jnp.stack((br*c - bphi*s, br*s + bphi*c, bz), axis=-1)


if __name__ == "__main__":
    points = jnp.array([[-8.2, 0.0, 0.0], [-8.2, 0.0, 1.0], [-8.2, 0.0, -1.0]])
    print("Positions [kpc]:\n", points)
    print("Bx, By, Bz [microgauss]:\n", gmf_cartesian(points))
