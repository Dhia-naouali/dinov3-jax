# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3

import math
from typing import Literal

import jax
import flax.linen as nn
import jax.numpy as jnp


class RopePositionEmbedding(nn.Module):
    embed_dim: int
    num_heads: int
    base: float | None = 100.0
    min_period: float | None = None
    max_period: float | None = None
    normalize_coords: Literal["min", "max", "separate"] = "separate"
    shift_coords: float | None = None
    jitter_coords: float | None = None
    rescale_coords: float | None = None
    dtype: jnp.dtype | None = None


    def setup(self):
        assert self.embed_dim % (4 * self.num_heads) == 0
        both_periods = self.min_period is not None and self.max_period is not None
        if (self.base is None and not both_periods) or (self.base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")
        D_head = self.embed_dim // self.num_heads

        if self.base is not None:
            periods = self.base ** (
                2. * jnp.arange(D_head // 4, dtype=self.dtype) / (D_head // 2.)
            )
        else:
            base = self.max_period / self.min_period
            exponents = jnp.linspace(0., 1., D_head // 4, dtype=self.dtype)  # [D//4] range [0, 1]
            periods = base**exponents  # range [1, max_period / min_period]
            periods = periods / base  # range [min_period / max_period, 1]
            periods = periods * self.max_period  # range [min_period, max_period]
            
        self.periods = self.variable("constants", "periods", lambda: periods)


    def __call__(self, *, H, W, deterministic=True, rng=None):
        d = {"dtype": self.dtype}
        
        # Prepare coords in range [-1, +1]
        match self.normalize_coords:
            case "max":
                max_HW = max(H, W)
                coords_h = jnp.arange(.5, H, **d) / max_HW  # [H]
                coords_w = jnp.arange(.5, W, **d) / max_HW  # [W]
            case "min":
                min_HW = max(H, W)
                coords_h = jnp.arange(.5, H, **d) / min_HW  # [H]
                coords_w = jnp.arange(.5, W, **d) / min_HW  # [W]
            case "separate":
                coords_h = jnp.arange(.5, H, **d) / H  # [H]
                coords_w = jnp.arange(.5, W, **d) / W  # [W]
            case _:
                raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")
        coords = jnp.stack(
            jnp.meshgrid(coords_h, coords_w, indexing="ij"), 
            axis=-1
        ).reshape(-1, 2)
        coords = 2. * coords - 1.
        
        if not deterministic:
            if rng is None and any(
                self.shift_coords, self.jitter_coords, self.rescale_coords
            ):
                raise ValueError("rng must be provided when deterministic=False and augmentations are enabled")
            
            rng, rng_shift, rng_jitter, rng_rescale = jax.random.split(rng, 4)

            if self.shift_coords is not None:
                shift_hw = jnp.random.uniform(
                    rng_shift, 
                    (2,), 
                    minval=-self.shift_coords,
                    maxval=self.shift_coords,
                    **d
                )
                coords += shift_hw[None, :]
            
            if self.jitter_coords is not None:
                jitter_max = jnp.log(self.jitter_coords)
                jitter_min = -jitter_max
                jitter_hw = jnp.exp(jax.random.uniform(
                    rng_jitter, 
                    (2,), 
                    minval=jitter_min,
                    maxval=jitter_max
                    **d
                ))
                coords *= jitter_hw[None, :]
            
            if self.rescale_coords is not None:
                rescale_max = jnp.log(self.rescale_coords)
                rescale_min = -rescale_max
                rescale_hw = jnp.exp(jax.random.uniform(
                    rng_rescale,
                    (1,),
                    minval=rescale_min,
                    maxval=rescale_max,
                    **d
                ))
                coords *= rescale_hw

        angles = 2 * math.pi * coords[:, :, None] / self.periods.value[None, None, :]
        angles = angles.reshape(angles.shape[0], -1)
        angles = jnp.concatenate([angles, angles], axis=-1)
        
        sin = jnp.sin(angles).astype(self.dtype)
        cos = jnp.cos(angles).astype(self.dtype)
        return sin, cos