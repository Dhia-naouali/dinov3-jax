# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3

# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.



# Dtype utilities for JAX/Flax
import jax.numpy as jnp

def to_dtype(x, dtype):
    return x.astype(dtype)
