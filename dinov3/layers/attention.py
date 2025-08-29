import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import *


def rope_rotate_half(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)

def rope_apply(x, sin, cos):
    return x * cos + rope_rotate_half(x) * sin

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
        self.scale = self.head_dim **-.5
        self.qkv = nn.Dense(self.dim*3, use_bias=self.mask_k_bias)
        self.proj = nn.Dense(self.dim, use_bias=self.proj_bias)
        self.attn_drop = nn.Dropout(self.proj_drop)
        
    def __call__(self, x, attn_bias=None, rope=None):
        qkv = self.qkv(x)
        attn_v = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope)
        x = self.proj(attn_v)
        x = self.proj_drop(x)
        return x

    def apply_rope(self, q, k, rope):
        cos, sin = rope
        rdt = rope.dtype
        qdt = q.dtype
        kdt = k.dtype

        q = q.astype(rdt)
        k = k.astype(rdt)
        
        n = q.shape[-2]
        prefix = n - sin.shape[-2]
        assert prefix >= 0

        q_prefix = q[:, :, :prefix, :]
        k_prefix = k[:, :, :prefix, :]

        q = rope_apply(q_prefix[:, :, prefix:, :], sin, cos)
        q = jnp.concatenate([q_prefix, q], dim=-2)

        k = rope_apply(k_prefix[:, :, prefix:, :], sin, cos)
        k = jnp.concatenate([k_prefix, k], dim=-2)
        
        q = q.astype(qdt)
        k = k.astype(kdt)
        return q, k


    def compute_attention(self, qkv, attn_bias, rope):
        assert attn_bias is None
        b, n, d3 = qkv.shape
        
        qkv = qkv.reshape(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = jnp.split(qkv, 3, axis=2) # b, n, 1, h, hd
        q = jnp.squeeze(q, axis=2).transpose(0, 2, 1, 3) # b, h, n, hd
        k = jnp.squeeze(k, axis=2).transpose(0, 2, 1, 3) # b, h, n, hd
        v = jnp.squeeze(v, axis=2).transpose(0, 2, 1, 3) # b, h, n, hd
        
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)



        x = nn.dot_product_attention(q, k, v, dropout_rate=self.attn_drop)
        x = jnp.transpose(0, 2, 1, 3)
        x = x.reshape(b, n, self.dim)
        return x
    