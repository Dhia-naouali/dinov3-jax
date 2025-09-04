# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3

import flax.linen as nn


class LayerScale(nn.Module):
    dim: int
    init_values: float = 1e-5
    inplace: bool = False

    def setup(self):
        self.gamma = self.param("gamma", nn.initializers.constant(self.init_values), (self.dim,))

    def __call__(self, x):
        return x * self.gamma