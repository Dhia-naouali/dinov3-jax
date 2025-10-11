# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3

# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.


import logging
from typing import List
from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn

logger = logging.getLogger("dinov3")


def drop_path(x, drop_prob=0., deterministic=True, rng=None):
    if drop_prob == 0. or deterministic:
        return x
    
    if rng is None:
        raise ValueError("no rng key passed with deterministic = True")
    
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_array = keep_prob + jnp.random.uniform(rng, shape, dtype=x.dtype)
    random_array = jnp.floor(random_array)
    return (x / keep_prob) * random_array

class DropPath(nn.Module):
    drop_prob: float = None
    
    def __call__(self, x, deterministic=True):
        return drop_path(
            x, 
            self.drop_prob, 
            deterministic=deterministic, 
            rng=self.make_rng("drop_path")
        )


class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.

    Source: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """
    
    dim: int
    drop_path=0.
    layer_scale_init_value=1e-6
    
    def setup(self):
        self.dwconv = nn.Conv(
            self.dim, 
            kernel_size=(7, 7), 
            padding="SAME", # sickens me
            feature_group_count=self.dim
        )
        self.norm = LayerNorm(eps=1e-6)
        self.pwconv1 = nn.Dense(self.dim*4)
        self.act = nn.gelu()
        self.pwconv2 = nn.Dense(self.dim)
        self.gamma = self.param(
            "gamma",
            nn.initializers.ones,
            (self.dim,)
        ) if self.layer_scale_init_value else None
        self.drop_path = DropPath(self.drop_path) if self.drop_path > 0. else nn.Identity()
    
    
    def __call__(self, x):
        raise Exception("fix shapes: using H, W, C")
        input_ = x
        x = self.dwconv(x)
        x = x.transpose(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(0, 3, 1, 2)
        x = input_ + self.drop_path(x)
        return x
    
class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).

    Source: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    eps: float = 1e-6
    data_format: str = "channels_last"
    
    @nn.compact
    def __call__(self, x):
        normalized_shape = (x.shape[-1],)
        assert self.data_format == "channels_last"
        self.weight = self.param(
            "weight",
            nn.initializers.ones,
            normalized_shape
        )
        self.bias = self.param(
            "bias",
            nn.initializers.zeros,
            normalized_shape
        )
        
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean((x - mean), axis=-1, keepdims=True) # mean already computed
        x = (x - mean) / (var + self.eps)
        return self.weight * x + self.bias


class ConvNeXt(nn.Module):
    r"""
    Code adapted from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.pyConvNeXt

    A PyTorch impl of : `A ConvNet for the 2020s`  -
        https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        patch_size (int | None): Pseudo patch size. Used to resize feature maps to those of a ViT with a given patch size. If None, no resizing is performed
    """

    # original ConvNeXt args
    depths: List[int] = [3, 3, 9, 3]
    dims: List[int] = [96, 192, 384, 768]
    drop_path_rate: float = 0.
    layer_scale_init_value: float = 1e-6
    # Dino args
    patch_size: int | None = None
    
    def setup(self):
        #################### ConvNeXt original init ####################
        self.downsample_layers = []
        stem = nn.Sequential(
            nn.Conv(
                self.dims[0],
                kernel_size=(4, 4),
                strides=(4, 4)
            ),
            LayerNorm()
        )
        self.downsample_layers.append(stem)
        
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(),
                nn.Conv(
                    self.dims[i+1],
                    kernel_size=(2, 2),
                    strides=(2, 2)
                )
            )
            self.downsample_layers.append(downsample_layer)
            
        self.stages = []
        dp_rates = [x for x in jnp.linspace(0, self.drop_path_rate, jnp.sum(jnp.array(self.depths)))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=self.dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=self.layer_scale_init_value
                    ) for j in range(self.depths[i])
                ]
            )
            self.stages.append(stage)
            cur += self.depths[i]
        
        self.norm = nn.LayerNorm(epsilon=1e-6)
        #########################################################
        #################### Dino adaptation ####################
        self.head = nn.Identity()
        self.embed_dim = self.dims[-1]
        self.embed_dims = self.dims
        self.n_blocks = len(self.downsample_layers)
        self.chunked_blocks = False
        self.n_storage_tokens = 0
        
        self.norms = [nn.Identity() for _ in range(3)]
        self.norms.append(self.norm)
        self.input_pad_size = 4
    
    
    def forward_features(self, x, masks=None):
        if isinstance(x, jnp.array):
            return self.forward_features_list([x], [masks])[0] # a77a 3
        else:
            return self.forward_features_list(x, masks)
    
    def forward_features_list(self, x_list, masks_list):
        output = []
        for x, masks in zip(x_list, masks_list):
            h, w = x.shape[-3:-1] # ahaaaa
            for i in range(4):
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)
            x_pool = x.mean(axis=(-3, -2)) # ahaaaa 2
            x = jnp.reshape(x.shape[0], -1, x.shape[-1])
            
            # n, hw+1, c
            x_norm = self.norm(jnp.concatenate([x_pool[:, None, :], x] axis=1))
            output.append({
                "x_norm_clstoken": x_norm[:, 0],
                "x_storage_tokens": x_norm[:, 1 : self.n_storage_tokens + 1],
                "x_norm_patchtokens": x_norm[:, self.n_storage_tokens + 1 :],
                "x_prenorm": x,
                "masks": masks,
            })
        return output


    def __call__(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        
        return self.head(ret["x_norm_clstoken"])
    
    
    def _get_intermediate_layers(self, x, n=1):
        h, w = x.shape[-3: -1]
        output, total_block_len = [], len(self.downsample_layers)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i in range(total_block_len):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in blocks_to_take:
                x_pool = x.mean(axis=(-3, -2))
                x_patches = x
                if self.patch_size is not None:
                    x_patches = jax.image.resize(
                        x,
                        shape=(x.shape[0], h // self.patch_size, w // self.patch_size, x.shape[-1]),
                        method="bilinear"
                    )
                output.append([
                    x_pool, x_patches
                ])
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output
    
    
    def get_intermediate_layers(
        self, x, n=1, reshape=False, return_class_token=False, norm=True
    ):
        outputs = self._get_intermediate_layers(x, n)
        if norm:
            nhwc_shapes = [out[-1].shape for out in outputs]
            if isinstance(n, int):
                norms = self.norms[-n]
            else:
                norm = [self.norms[i] for i in n]
            outputs = [
                (
                    norm(cls_token),
                    norm(patches.reshape(patches.shape[0], -1, patches.shape[-1]))
                ) for (cls_token, patches), norm in zip(outputs, norms)
            ]
            if reshape:
                outputs = [
                    (cls_token, patches.transpose(0, 2, 1).reshape(*nhwc))  # To-Do: shape mess fix
                    for (cls_token, patches), nhwc in zip(outputs, nhwc_shapes)
                ]
        elif not reshape:
            outputs = [
                (
                    cls_token, 
                    patches.reshape(patches.shape[0], -1, patches.shape[-1])
                ) for cls_token, patches in outputs
            ]
        class_tokens = [out[0] for out in outputs]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        
        return tuple(outputs)


convnext_sizes = {
    "tiny": dict(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
    ),
    "small": dict(
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
    ),
    "base": dict(
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
    ),
    "large": dict(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
    ),
}


def get_convnext_arch(arch_name):
    size_dict = None
    query_sizename = arch_name.split("_")[1]
    try:
        size_dict = convnext_sizes[query_sizename]
    except KeyError:
        raise NotImplementedError("didn't recognize vit size string")

    return partial(
        ConvNeXt,
        **size_dict,
    )