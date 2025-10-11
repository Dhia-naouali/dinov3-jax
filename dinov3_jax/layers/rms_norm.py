# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3

import jax
import jax.numpy as jnp
import flax.linen as nn

class RMSNorm(nn.Module):
    eps: float = 1e-5
    
    @nn.compact
    def __call__(self, x):
        self.weight = self.param("weight", nn.initializers.ones, (x.shape[-1],))
        norm = jax.lax.rsqrt(
            jnp.mean(
                jnp.square(x.astype(jnp.float)), 
                axis=-1, 
                keepdims=True
            ) + self.eps
        )
        
        x = x * norm.astype(x.dtype)
        return self.weight * x