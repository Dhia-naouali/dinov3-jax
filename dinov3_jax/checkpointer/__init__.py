# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3

# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.


from .checkpointer import (
    CheckpointRetentionPolicy,
    cleanup_checkpoint,
    find_latest_checkpoint,
    keep_checkpoint_copy,
    keep_last_n_checkpoints,
    load_checkpoint,
    save_checkpoint
)