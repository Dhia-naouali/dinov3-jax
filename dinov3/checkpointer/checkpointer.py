# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3

import logging
import shutil
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import List, Sequence, Set

import jax
import jax.numpy as jnp
from orbas.checkpoint import ocp
import flax.serialization as fserial

from .dinov3 import distributed

logger = logging.getLogger("dinov3")

class CheckpointRetentionPolicy(Enum):
    ALL = "all"  # keep all checkpoints
    BEST = "best"
    LAST = "last"
    LAST_AND_BEST = "last_and_best"
    NONE = "none"  # do not keep any checkpoints

    @property
    def keep_filters(self):
        if self == CheckpointRetentionPolicy.LAST:
            return set(["final"])
        if self == CheckpointRetentionPolicy.BEST:
            return set(["best"])
        if self == CheckpointRetentionPolicy.LAST_AND_BEST:
            return set(["final", "best"])
        if self == CheckpointRetentionPolicy.ALL:
            return set()
        return set()


    @property
    def max_to_keep(self):
        if self == CheckpointRetentionPolicy.ALL:
            return None
        return 1

def save_checkpoint(
    ckpt_dir,
    *,
    iteration,
    model,
    optimizer,
    overwrite=True,
    process_group=None,
    **others # to keep ?
):
    rank = distributed.get_rank()
    
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir_exists = [ckpt_dir.exists() if rank == 0 else None]
    src_rank = 0
    if process_group is None:
        src_rank = 