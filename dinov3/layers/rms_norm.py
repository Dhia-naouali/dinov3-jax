import jax
import jax.numpy as jnp
import flax.linen as nn

class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-5
    
    @nn.compact
    def __call__(self, x):
        self.weight = self.param("weight", nn.initializers.ones, (self.dim,))
        norm = jax.lax.rsqrt(
            jnp.mean(
                jnp.square(x.astype(jnp.float)), 
                axis=-1, 
                keepdims=True
            ) + self.eps
        )
        
        x = x * norm.astype(x.dtype)
        return self.weight * x


    # def reset_parameters(self):
    #     self.weight = jnp.ones_like(self.weight)