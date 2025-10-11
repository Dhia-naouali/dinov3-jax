# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3

# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.


# this file was imported as is from the original repo since no changes are needed to adapt to flax/jax


from .ade20k import ADE20K
from .coco_captions import CocoCaptions
from .image_net import ImageNet
from .image_net_22k import ImageNet22k