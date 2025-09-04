# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3

import flax.linen as nn
from typing import Union, Tuple, Callable

def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    
    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    img_size: Union[int, Tuple[int, int]] = 224
    patch_size: Union[int, Tuple[int, int]] = 16
    # in_chans: int = 3
    embed_dim: int = 768
    norm_layer: Callable | None = None
    flatten_embedding: bool = True
    
    def setup(self):
        self._img_size = make_2tuple(self.img_size)
        self._patch_size = make_2tuple(self.patch_size)
        self.patches_resolution = (
            self._img_size[0] // self._patch_size[0],
            self._img_size[1] // self._patch_size[1],
        )
        
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.proj = nn.Conv(
            self.embed_dim, 
            kernel_size=self._patch_size, 
            strides=self._patch_size
        )
        self.norm = self.norm_layer() if self.norm_layer else lambda x: x
        
    def __call__(self, x):
        b, h, w, c = x.shape
        patch_h, patch_w = self._patch_size
        assert h % patch_h == 0, f"input image height {h} is not a multiple of patch height {patch_h}"
        assert w % patch_w == 0, f"input image width {w} is not a multiple of patch width {patch_w}"
        
        x = self.proj(x) # b, h, w, d
        h, w = x.shape[-3:-1]
        if self.flatten_embedding:
            x = x.reshape(b, -1, self.embed_dim) # b, n, d
        return self.norm(x)
    
    