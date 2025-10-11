# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3


import jax


def is_enabled():
    return jax.device_count() > 1


def get_rank():
    return 0 # single host


def get_world_size():
    return jax.device_count()