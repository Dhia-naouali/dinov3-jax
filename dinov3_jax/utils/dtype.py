# Dtype utilities for JAX/Flax
import jax.numpy as jnp

def to_dtype(x, dtype):
    return x.astype(dtype)
