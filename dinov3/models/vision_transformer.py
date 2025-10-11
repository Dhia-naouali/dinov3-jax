# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3

import logging
from functools import partial
from typing import *

import jax
import flax.linen as nn
import jax.numpy as jnp

# from dinov3.utils import named_apply # pytree map ?
from dinov3.layers import (
    LayerScale, 
    Mlp, 
    PatchEmbed, 
    RMSNorm, 
    RopePositionEmbedding, 
    SelfAttentionBlock, 
    SwiGLUFFN
)
from dinov3.fsdp.utils import fsdp_wrapper

logger = logging.getLogger("dinov3")

ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, epsilon=1e-6),
    "layernormbf16": partial(nn.LayerNorm, epsilon=1e-5),
    "rmsnorm": RMSNorm
}

dtype_dict = {
    "fp32": jnp.float32,
    "fp16": jnp.float16,
    "bf16": jnp.float16
}

# def init_weights_vit(module, name = ""):
#     if isinstance(module, nn.Dense):



class DinoVisionTransformer(nn.Module):
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    pos_embed_rope_base: float = 100.0
    pos_embed_rope_min_period: float | None = None
    pos_embed_rope_max_period: float | None = None
    pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate"
    pos_embed_rope_shift_coords: float | None = None
    pos_embed_rope_jitter_coords: float | None = None
    pos_embed_rope_rescale_coords: float | None = None
    pos_embed_rope_dtype: str = "bf16"
    embed_dim: int = 768
    n_blocks: int = 12 # depth
    num_heads: int = 12
    ffn_ratio: float = 4.0
    qkv_bias: bool = True
    drop_path_rate: float = 0.0
    layerscale_init: float | None = None
    norm_layer: str = "layernorm"
    ffn_layer: str = "mlp"
    ffn_bias: bool = True
    proj_bias: bool = True
    n_storage_tokens: int = 0
    mask_k_bias: bool = False
    untie_cls_and_patch_norms: bool = False
    untie_global_and_local_cls_norm: bool = False
    
    def setup(self):
        if jax.device_count() > 1:
            self.fsdp: Callable = partial(fsdp_wrapper, axis_name="dp")
        else:
            self.fsdp = lambda x: x
        norm_layer_cls = norm_layer_dict[self.norm_layer]
        self.num_features = self.embed_dim
        
        self.patch_embed = self.fsdp(PatchEmbed)(
            img_size=self.img_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            flatten_embedding=False
        )
        
        self.cls_token = self.param(
            "cls_token", 
            nn.initializers.normal(.02), 
            (1, 1, self.embed_dim)
        )
        
        if self.n_storage_tokens > 0:
            self.storage_tokens = self.param(
                "sotrage_tokens",
                nn.initializers.normal(.02),
                (1, self.n_storage_tokens, self.embed_dim)
            )
        logger.info(f"using base={self.pos_embed_rope_base} for rope new")
        logger.info(f"using min_period={self.pos_embed_rope_min_period} for rope new")
        logger.info(f"using max_period={self.pos_embed_rope_max_period} for rope new")
        logger.info(f"using normalize_coords={self.pos_embed_rope_normalize_coords} for rope new")
        logger.info(f"using shift_coords={self.pos_embed_rope_shift_coords} for rope new")
        logger.info(f"using rescale_coords={self.pos_embed_rope_rescale_coords} for rope new")
        logger.info(f"using jitter_coords={self.pos_embed_rope_jitter_coords} for rope new")
        logger.info(f"using dtype={self.pos_embed_rope_dtype} for rope new")

        self.rope_embed = RopePositionEmbedding(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            base=self.pos_embed_rope_base,
            min_period=self.pos_embed_rope_min_period,
            max_period=self.pos_embed_rope_max_period,
            normalize_coords=self.pos_embed_rope_normalize_coords,
            shift_coords=self.pos_embed_rope_shift_coords,
            jitter_coords=self.pos_embed_rope_jitter_coords,
            rescale_coords=self.pos_embed_rope_rescale_coords,
            # dtype=dtype_dict[self.pos_embed_rope_dtype],
        )
        logger.info(f"using {self.ffn_layer} layer as FFN")
        ffn_layer_cls = ffn_layer_dict[self.ffn_layer]
        ffn_ratio_sequence = [self.ffn_ratio] * self.n_blocks
        self.blocks = [
            self.fsdp(SelfAttentionBlock)(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=self.qkv_bias,
                proj_bias=self.proj_bias,
                ffn_bias=self.ffn_bias,
                drop_path=self.drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.gelu,
                ffn_layer=ffn_layer_cls,
                init_values=self.layerscale_init,
                mask_k_bias=self.mask_k_bias,
            ) for i in range(self.n_blocks)
        ]
        
        self.chunked_blocks = False
        self.norm = norm_layer_cls()
        
        if self.untie_cls_and_patch_norms:
            self.cls_norm = norm_layer_cls()
        else:
            self.cls_norm = None
        
        if self.untie_global_and_local_cls_norm:
            self.local_cls_norm = norm_layer_cls()
        else:
            self.local_cls_norm = None
        
        self.head = lambda x: x
        self.mask_token = self.param(
            "mask_token",
            nn.initializers.zeros,
            (1, self.embed_dim)
        )
    
    def prepare_tokens_with_masks(self, x, masks=None):
        x = self.patch_embed(x)
        B, H, W, C = x.shape
        x = x.reshape(B, -1, C)
        
        if masks is not None:

            x = jnp.where(
                masks[..., None],
                self.mask_token.astype(x.dtype)[None, ...],
                x
            )
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token # prolly for broadcasting
        
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = jnp.empty(
                (1, 0, cls_token.shape[-1]),
                dtype=cls_token.dtype
            )
        
        x = jnp.concatenate([
            jnp.broadcast_to(cls_token, (B,) + cls_token.shape[1:]),
            jnp.broadcast_to(storage_tokens, (B,) + storage_tokens.shape[1:]),
            x
        ], axis=1)
        
        return x, (H, W)
    
    def forward_features_list(self, x_list, masks_list, deterministic):
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x); rope.append(hw_tuple)
            
        for block in self.blocks:
            if self.rope_embed is not None:
                rope_sincos = [self.rope_embed(H=H, W=W) for H, W in rope]
            else:
                rope_sincos = [None for _ in rope]
            x = block(x, rope_sincos, deterministic=deterministic)
        
        all_x = x
        output = []


        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and not deterministic and idx == 1:
                    x_norm_cls_reg = self.local_cls_norm(x[:, : self.n_storage_tokens + 1])
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(x[:, : self.n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = self.norm(x[:, : self.n_storage_tokens + 1])
                
                x_norm_patch = self.norm(x[:, self.n_storage_tokens + 1 :])
            else:
                x_norm = self.norm(x)
                x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x,
                    "masks": masks
                }
            )
        
        return output
    
    
    def forward_features(self, x, masks=None, deterministic=True):
        # try:
        if isinstance(x, jnp.ndarray):
            return self.forward_features_list([x], [masks], deterministic=deterministic)[0] # a77a 2
        else:
            return self.forward_features_list(x, masks, deterministic=deterministic)
        # except Exception as e:
            # this is a bottleneck where to catch nan values during debugging
            # print(e)
            # import IPython; IPython.embed()

    
    def _get_intermediate_layers_not_chunked(self, x, n=1, deterministic=True):
        x, (H, W) = self.prepare_tokens_with_masks(x)
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        
        for i, block in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            
            x = block(x, rope_sincos, deterministic=deterministic)
            if i in blocks_to_take:
                output.append(x)
        
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output
    
    def get_intermediate_layers(
        self, x, *, n=1, reshape=False, return_class_token=False, return_extra_tokens=False, norm=True
    ):
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, :, self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1, :])
                    outputs_normed.append(jnp.concatenate([x_norm_cls_reg, x_norm_patch], axis=1))
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1, :] for out in outputs]
        
        if reshape:
            B, _, h, w = x.shape
            outputs = [
                out.reshape(B, h // self.patch_size, w // self.patch_size, -1).transposek(0, 3, 1, 2)
                for out in outputs
            ]
        
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        elif return_class_token and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens))
        
    
    def __call__(self, *args, is_training=False, deterministic=True, **kwargs):
        # is_traingin is used to determin whether all forward pass results shoudl be returned or just the cls token
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        
        return self.head(ret["x_norm_clstoken"])



def vit_small(patch_size=16, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        n_blocks=12,
        num_heads=6,
        ffn_ratio=4,
        # nvm these people
        # embed_dim=128, 
        # n_blocks=2,
        # num_heads=2,
        # ffn_ratio=1,
        **kwargs
    )



def vit_base(patch_size=16, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        n_blocks=12,
        num_heads=12,
        ffn_ratio=4,
        **kwargs,
    )


def vit_large(patch_size=16, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        n_blocks=24,
        num_heads=16,
        ffn_ratio=4,
        **kwargs,
    )


def vit_so400m(patch_size=16, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1152,
        n_blocks=27,
        num_heads=18,
        ffn_ratio=3.777777778,
        **kwargs,
    )


def vit_huge2(patch_size=16, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1280,
        n_blocks=32,
        num_heads=20,
        ffn_ratio=4,
        **kwargs,
    )


def vit_giant2(patch_size=16, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        n_blocks=40,
        num_heads=24,
        ffn_ratio=4,
        **kwargs,
    )
    

def vit_7b(patch_size=16, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=4096,
        n_blocks=40,
        num_heads=32,
        ffn_ratio=3,
        **kwargs,
    )