"""Periodic correlated lognormal source on interior XY points."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import nifty8.re as jft

def make_source_prior(shape, dx):
    """Return a positive NIFTy model, including all spectrum hyperparameters."""
    cfm = jft.CorrelatedFieldMaker("source_")
    # Log-field centered near zero, with modest uncertainty in its mean.
    cfm.set_amplitude_total_offset(offset_mean=0.0, offset_std=(0.7, 0.3))
    cfm.add_fluctuations(
        shape, distances=(dx, dx),
        fluctuations=(0.8, 0.4), loglogavgslope=(-4.0, 0.5),
        flexibility=(0.5, 0.1), asperity=(0.5, 0.1),
        prefix="xy_", non_parametric_kind="power",
    )
    field = cfm.finalize()
    return jft.Model(lambda latent: jnp.exp(field(latent)),
                     domain=field.domain, init=field.init)
