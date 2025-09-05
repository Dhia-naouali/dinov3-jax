# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3

from typing import Callable

import jax.numpy as jnp
import flax.linen as nn

def rope_rotate_half(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def rope_apply(x, sin, cos):
    return x * cos + rope_rotate_half(x) * sin


class LinearKMaskedBias(nn.Module):
    features: int
    use_bias: bool = False
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        assert self.features % 3 == 0        
        kernel = self.param(
            "kernel",
            self.kernel_init,
            (x.shape[-1], self.features)
        )
        
        out = x @ kernel

        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,))
            bias_mask = self.variable("constants", "bias_mask", lambda: jnp.full((self.features,), jnp.nan))
            out += bias * bias_mask.value

        return  out



class SelfAttention(nn.Module):
    dim: int
    num_heads: int = 8
    qkv_bias: bool = False
    proj_bias: bool = True
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    mask_k_bias: bool = False

    def setup(self):
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -.5
        linear_class = LinearKMaskedBias if self.mask_k_bias else nn.Dense

        self.qkv = linear_class(self.dim * 3, use_bias=self.qkv_bias)
        self.attn_drop = nn.Dropout(self.attn_drop)
        self.proj = nn.Dense(self.dim, use_bias=self.proj_bias)
        self.proj_drop = nn.Dropout(self.proj_drop)


    def apply_rope(self, q, k, rope):
        sin, cos = rope
        rdt = sin.dtype
        qdt, kdt = q.dtype, k.dtype

        q, k = q.astype(rdt), k.astype(rdt)
        q, k = [jnp.transpose(t, (0, 2, 1, 3)) for t in [q, k]]        
        
        prefix = q.shape[-2] - sin.shape[-2]
        assert prefix >= 0

        q_prefix = q[:, :, :prefix, :]
        k_prefix = k[:, :, :prefix, :]

        q = rope_apply(q[:, :, prefix:, :], sin, cos)
        k = rope_apply(k[:, :, prefix:, :], sin, cos)

        q = jnp.concatenate([q_prefix, q], axis=-2).astype(qdt)
        k = jnp.concatenate([k_prefix, k], axis=-2).astype(kdt)

        q, k = [jnp.transpose(t, (0, 2, 1, 3)) for t in [q, k]]
        return q, k        


    def __call__(self, x, attn_bias=None, rope=None, deterministic=True):
        qkv = self.qkv(x)
        attn_v = self.compute_attention(
            qkv, 
            attn_bias=attn_bias, 
            rope=rope,
            deterministic=deterministic
        )
        x = self.proj(attn_v)
        x = self.proj_drop(x, deterministic=deterministic)
        return x
    

    def compute_attention(self, qkv, attn_bias, rope=None, deterministic=True):
        assert attn_bias is None
        B, N, _ = qkv.shape
        
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = jnp.moveaxis(qkv, 2, 0) # b, n, h, hd
        
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)

        x = nn.dot_product_attention(q, k, v, deterministic=deterministic)
        x = x.reshape(B, N, self.dim)
        return x


    # def apply_list(self, x_list, attn_bias=None, rope_list=None, deterministic=True):
    #     assert len(x_list) == len(rope_list)
    #     x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
    #     qkv_flat = self.qkv(x_flat)
    #     qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)
        
        
    #     def apply_(x, rope):
    #         qkv = self.qkv(x)
    #         return self.compute_attnetion(qkv, rope=rope, deterministic=deterministic)

    #     return jax.vmap(apply_)(x_list, rope_list)


class CausalSelfAttention(nn.Module):
    dim: int
    num_heads: int = 8
    qkv_bias: bool = False
    proj_bias: bool = True
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    init_attn_std: float = None
    init_proj_std: float = None
    factor: float = 1.
    
    def setup(self):
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -.5

        # initializers
        self.init_attn_std = self.init_attn_std or self.dim**-.5
        self.init_proj_std = self.init_proj_std or self.init_attn_std * self.factor
                
        self.qkv = nn.Dense(
            self.dim*3, 
            use_bias=self.qkv_bias,
            kernel_init=nn.initializers.normal(self.init_attn_std),
            bias_init=nn.initializers.zeros
        )
        self.proj = nn.Dense(
            self.dim, 
            use_bias=self.proj_bias,
            kernel_init=nn.initializers.normal(self.init_proj_std),
            bias_init=nn.initializers.zeros
        )
        
        self.proj_drop = nn.Dropout(self.proj_drop)
    
    
    def __call__(self, x, is_causal=True, deterministic=True):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = jnp.moveaxis(qkv, 2, 0) # b, n, h, hd
        x = nn.dot_product_attention(
            q, k, v,
            mask=nn.make_causal_mask(jnp.ones((B, N))) if is_causal else None,
            dropout_rate=self.attn_drop,
            deterministic=deterministic
        )
        x = x.reshape(B, N, -1)
        x = self.proj_drop(self.proj(x), deterministic=deterministic)
        return x