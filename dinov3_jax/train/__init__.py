# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3

from .multidist_meta_arch import MultiDistillationMetaArch
from .ssl_meta_arch import SSLMetaArch
from .train import get_args_parser, main # commenting this temporarly