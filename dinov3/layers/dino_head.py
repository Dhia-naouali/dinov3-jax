# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3

import jax
import flax.linen as nn
import jax.numpy as jnp
from functools import partial


def _build_mlp(
    nlayers, 
    bottleneck_dim,
    hidden_dim=None,
    use_bn=False,
    bias=True
):
    linear = partial(
        nn.Dense, 
        use_bias=bias, 
        kernel_init=jax.nn.initializers.truncated_normal(
            stddev=.02, 
            lower=-1., 
            upper=1.
        ), # default values in torch.nn.init.trunc_normal_
        bias_init=nn.initializers.zeros
        )

    if nlayers == 1:
        return linear(bottleneck_dim)
    
    layers = []
    for _ in range(nlayers - 1):
        layers.append(linear(hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm())
        layers.append(nn.gelu)
    layers.append(linear(bottleneck_dim))
    return nn.Sequential(layers)


class DINOHead(nn.Module):
    in_dim: int
    out_dim: int
    use_bn: bool = False
    nlayers: int = 3
    hidden_dim: int = 2048
    bottleneck_dim: int = 256
    mlp_bias: bool = True


    def setup(self):
        self.mlp = _build_mlp(
            self.nlayers,
            # self.in_dim,
            self.bottleneck_dim,
            hidden_dim=self.hidden_dim,
            use_bn=self.use_bn,
            bias=self.mlp_bias,
        )
        self.last_layer = nn.Dense(
            self.out_dim, 
            use_bias=False,
            kernel_init=jax.nn.initializers.truncated_normal(
                stddev=.02, 
                lower=-1., 
                upper=1.
            ),
            bias_init=nn.initializers.zeros
        )


    def __call__(self, x, no_last_layer=False, only_last_layer=False):
        if not only_last_layer:
            x = self.mlp(x)
            eps = 1e-6 if x.dtype == jnp.float16 else 1e-12
            norm = jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True)
            x /= (norm + eps)
        if not no_last_layer:
            x = self.last_layer(x)
        return x