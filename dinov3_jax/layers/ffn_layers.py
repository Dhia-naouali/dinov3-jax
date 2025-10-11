# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import flax.linen as nn
from typing import Callable, Optional
from dinov3_jax.utils import cat_keep_shapes, uncat_with_shapes


class ListForwardMixin(object):
    def __call__(self, x):
        raise NotImplemented
    
    def forward_list(self, x_list):
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        x_flat = self(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)


class Mlp(nn.Module, ListForwardMixin):
    # in_features: int
    hidden_features: Optional[int] = None
    out_features: Optional[int] = None
    act_layer: Callable[..., nn.Module] = nn.gelu
    drop: float = 0.0
    use_bias: bool = True


    @nn.compact
    def __call__(self, x, deterministic=True):
        in_dim = x.shape[-1]
        x = nn.Dense(
            self.hidden_features or in_dim,
            use_bias=self.use_bias
        )(x)
        x = self.act_layer(x)
        x = nn.Dropout(self.drop)(x, deterministic=deterministic)

        x = nn.Dense(
            self.out_features or in_dim,
            use_bias=self.use_bias
        )(x)
        x = self.act_layer(x)
        x = nn.Dropout(self.drop)(x, deterministic=deterministic)
        return x


class SwiGLUFFN(nn.Module, ListForwardMixin):
    # in_features: int
    hidden_features: Optional[int] = None
    out_features: Optional[int] = None
    act_layer: Optional[Callable[..., nn.Module]] = None # placeholder
    drop: float = 0.0
    use_bias: bool = True
    align_to: int = 8
    
    def setup(self):
        # self.out_features = self.out_features or self.in_features
        # self.hidden_features = self.hidden_features or self.in_features
        d = int(self.hidden_features * 2 / 3)
        swiglu_hidden_features = d + (-d % self.align_to)
        self.w1 = nn.Dense(swiglu_hidden_features, use_bias=self.use_bias)
        self.w2 = nn.Dense(swiglu_hidden_features, use_bias=self.use_bias)
        self.w3 = nn.Dense(self.out_features, use_bias=self.use_bias)
        
    def __call__(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = nn.silu(x1) * x2
        return self.w3(hidden)