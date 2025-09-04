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
    bias_init: Callable = None

    @nn.compact
    def __call__(self, x):
        assert self.features % 3 == 0
        
        kernel = self.param(
            "kernel",
            self.kernel_init,
            (x.shape[-1], self.features)
        )
        
        if self.use_bias:
            bias = self.variable("constants", "bias", lambda: jnp.full((self.features,), jnp.nan))

        out = x @ kernel
        if self.use_bias:
            out += bias.value
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
        b, n, _ = qkv.shape
        
        qkv = qkv.reshape(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = jnp.split(qkv, 3, axis=2) # b, n, 1, h, hd
        q = jnp.squeeze(q, axis=2) # b, n, h, hd
        k = jnp.squeeze(k, axis=2) # b, n, h, hd
        v = jnp.squeeze(v, axis=2) # b, n, h, hd
        
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)

        x = nn.dot_product_attention(q, k, v, deterministic=deterministic)
        x = x.reshape(b, n, self.dim)
        return x


    # def apply_list(self, x_list, attn_bias, rope_list=None, deterministic=True):
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

    def setup(self):
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -.5
        self.qkv = nn.Dense(self.dim*3, use_bias=self.qkv_bias)
        self.proj = nn.Dense(self.dim, use_bias=self.proj_bias)
        self.proj_drop = nn.Dropout(self.proj_drop)



class CausalSelfAttention(nn.Module):
    def init_weights(
        self, init_attn_std: float | None = None, init_proj_std: float | None = None, factor: float = 1.0
    ) -> None:
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        nn.init.normal_(self.qkv.weight, std=init_attn_std)
        nn.init.normal_(self.proj.weight, std=init_proj_std)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, is_causal: bool = True) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0, is_causal=is_causal
        )
        x = x.transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x