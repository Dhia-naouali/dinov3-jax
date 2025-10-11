# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3

# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.


import jax
import jax.numpy as jnp
import flax.linen as nn

from typing import Callable, Optional

from dinov3_jax.utils import cat_keep_shapes, uncat_with_shapes
from .attention import SelfAttention, CausalSelfAttention
from .ffn_layers import Mlp
from .layer_scale import LayerScale



class SelfAttentionBlock(nn.Module):
    dim: int
    num_heads: int
    ffn_ratio: float = 4.0
    qkv_bias: bool = False
    proj_bias: bool = True
    ffn_bias: bool = True
    drop: float = 0.0
    attn_drop: float = 0.0
    init_values: float = None
    drop_path: float = 0.0
    act_layer: Callable[..., nn.Module] = nn.gelu
    norm_layer: Callable[..., nn.Module] = nn.LayerNorm
    attn_class: Callable[..., nn.Module] = SelfAttention
    ffn_layer: Callable[..., nn.Module] = Mlp
    mask_k_bias: bool = False

    def setup(self):
        self.norm1 = self.norm_layer()
        self.attn = self.attn_class(
            self.dim,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            proj_bias=self.proj_bias,
            attn_drop=self.attn_drop,
            proj_drop=self.drop,
            mask_k_bias=self.mask_k_bias,
        )
        self.ls1 = LayerScale(
            self.dim, init_values=self.init_values
        ) if self.init_values else lambda x: x
        
        self.norm2 = self.norm_layer()
        mlp_hidden_dim = int(self.dim * self.ffn_ratio)
        self.mlp = self.ffn_layer(
            hidden_features=mlp_hidden_dim,
            act_layer=self.act_layer,
            drop=self.drop,
            use_bias=self.ffn_bias,
        )
        
        self.ls2 = LayerScale(
            self.dim, init_values=self.init_values
        ) if self.init_values else lambda x: x
        self.sample_drop_ratio = self.drop_path
        
    @staticmethod
    def _maybe_index_rope(rope, indices):
        if rope is None:
            return None
        
        sin, cos = rope
        assert sin.ndim == cos.ndim

        if sin.ndim == 4:
            return sin[indices], cos[indices]
        
        return sin, cos
    
    def _apply(self, x, rope=None, deterministic=True):
        """
        x -> norm -> attention -> scale  |
        [res]--------------------------> + -> norm -> mlp -> scale |
                                       [res]---------------------> + -> out 
        """
        b, _, _ = x.shape
        sample_subset_size = jnp.maximum(
            jnp.floor(b * (1 - self.sample_drop_ratio)).astype(jnp.int32),
            1
        )
        residual_scale_factor = b / sample_subset_size
        
        if not deterministic and self.sample_drop_ratio > 0.:
            indices_1 = jax.random.permutation(
                self.make_rng("drop_path"), b
            )[:sample_subset_size]

            x_subset_1 = x[indices_1]
            rope_subset = self._maybe_index_rope(rope, indices_1)
            residual_1 = self.attn(self.norm1(x_subset_1), rope=rope_subset, deterministic=deterministic)
            
            x_attn = x.at[indices_1].add(
                self.ls1(residual_1) * residual_scale_factor
            )
            
            
            indices_2 = jax.random.permutation(
                self.make_rng("drop_path"), b
            )[:sample_subset_size]
            
            x_subset_2 = x_attn[indices_2]
            residual_2 = self.mlp(self.norm2(x_subset_2))
            
            x_ffn = x_attn.at[indices_2].add(
                self.ls2(residual_2) * residual_scale_factor
            )
        else:
            x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
            x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))
        
        return x_ffn
    
    def _forward_list(self, x_list, rope_list=None, deterministic=True):
        b_list = [x.shape[0] for x in x_list]
        sample_subset_sizes = [
            jnp.maximum(
                jnp.floor(b * (1 - self.sample_drop_ratio)).astype(jnp.int32),
                1
            ) for b in b_list
        ]
        
        residual_scale_factors = [
            b / sample_subset_size for b, sample_subset_size in zip(
                b_list, sample_subset_sizes
            )
        ]
        

        if not deterministic and self.sample_drop_ratio > 0.:
            indices_1_list = [
                jax.random.permutation(
                    self.make_rng("drop_path"), b
                )[:sample_subset_size]
                for b, sample_subset_size in zip(b_list, sample_subset_sizes)
            ]
            
            x_subset_1_list = [x[indices_1] for x, indices_1 in zip(x_list, indices_1_list)]
            
            if rope_list is not None:
                rope_subset_list = [
                    self._maybe_index_rope(rope, indices_1) for rope, indices_1 in zip(
                        rope_list, indices_1_list
                    )
                ]
            else: 
                rope_subset_list = rope_list
            
            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_1_list)
            norm1 = uncat_with_shapes(self.norm1(flattened), shapes, num_tokens)
            residual_1_list = self.attn.apply_list(norm1, rope_list=rope_subset_list, deterministic=deterministic)
            
            x_attn_list = [
                x.at[indices_1].add(self.ls1(residual_1) * residual_scale_factor)
                for x, residual_1, indices_1, residual_scale_factor in zip(
                    x_list, residual_1_list, indices_1_list, residual_scale_factors
                )
            ]
            
            indices_2_list = [
                jax.random.permutation(
                    self.make_rng("drop_path"), b
                )[:sample_subset_size] for b, sample_subset_size in zip(
                    b_list, sample_subset_sizes
                )
            ]
            x_subeset_2_list = [
                x[indices_2] for x, indices_2 in zip(
                    x_attn_list, indices_2_list
                )
            ]
            
            flattened, shapes, num_tokens = cat_keep_shapes(x_subeset_2_list)
            norm2_flat = self.norm2(flattened)
            norm2_list = uncat_with_shapes(norm2_flat, shapes, num_tokens)
            residual_2_list = self.mlp.apply_list(norm2_list)
            
            x_ffn = [
                x_attn.at[indices_2].add(
                    self.ls2(residual_2) * residual_scale_factor
                )for x_attn, residual_2, indices_2, residual_scale_factor in zip(
                    x_attn_list, residual_2_list, indices_2_list, residual_scale_factors
                )
            ]
        else:
            x_out = []
            for x, rope in zip(x_list, rope_list):
                x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope, deterministic=deterministic))
                x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))
                x_out.append(x_ffn)
            x_ffn = x_out
        
        return x_ffn

    def __call__(self, x_or_x_list, rope_or_rope_list=None, deterministic=True):
        if isinstance(x_or_x_list, jnp.ndarray):
            # a77a
            return self._apply_list([x_or_x_list], rope_list=[rope_or_rope_list], deterministic=deterministic)[0]
        elif isinstance(x_or_x_list, list):
            if rope_or_rope_list is None:
                rope_or_rope_list = [None for _ in rope_or_rope_list]
            
            return self._forward_list(x_or_x_list, rope_list=rope_or_rope_list, deterministic=deterministic)
        raise AssertionError




class CausalSelfAttentionBlock(nn.Module):
    dim: int
    num_heads: int
    ffn_ratio: float = 4.0
    ls_init_value: Optional[float] = None
    is_causal: bool = True
    act_layer: Callable = nn.gelu
    norm_layer: Callable = nn.LayerNorm
    dropout_prob: float = 0.0
    init_attn_std: float | None = None
    init_proj_std: float | None = None
    init_fc_std: float | None = None
    factor: float = 1.0
    
    
    def setup(self):
        init_attn_std = self.init_attn_std or (self.dim**-.5)
        init_proj_std = self.init_proj_std or init_attn_std * self.factor
        init_fc_std = self.init_fc_std or (2*self.dim) ** -.5
        
        ################################################################
        ############################# init #############################
        ################################################################
        
        self.ls1 = LayerScale(self.dim, init_values=self.ls_init_value) if self.ls_init_value else lambda x: x
        self.attention_norm = self.norm_layer()
        self.attention = CausalSelfAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            attn_drop=self.dropout_prob
        )
        self.ffn_norm = self.norm_layer()
        ffn_hidden_dim = jnp.floor(self.dim * self.ffn_ratio).astype(jnp.int32)
        self.feed_forward = Mlp(
            # in_features=self.dim,
            hidden_features=ffn_hidden_dim,
            act_layer=self.act_layer
        )
        
        self.ls2 = LayerScale(self.dim, init_values=self.ls_init_value) if self.ls_init_value else lambda x: x
        
    def __call__(self, x):
        x_attn = x + self.ls1(self.attention(self.attention_norm(x), self.is_causal))
        x_ffn = x_attn + self.ls2(self.feed_forward(self.ffn_norm(x_attn)))
        return x_ffn