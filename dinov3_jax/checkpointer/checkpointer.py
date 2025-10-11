# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3

# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.


import jax
import logging
import shutil
import tempfile
import subprocess
from enum import Enum
from pathlib import Path
from typing import List, Sequence, Set

import orbax.checkpoint as ocp


logger = logging.getLogger("dinov3")

class CheckpointRetentionPolicy(Enum):
    ALL = "all"
    BEST = "best"
    LAST = "last"
    LAST_AND_BEST = "last_and_best"
    NONE = "none"

    @property
    def keep_filters(self) -> Set[str]:
        """Files that match these patterns are not deleted by cleanup"""
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
    def max_to_keep(self) -> int | None:
        """
        maximum "periodic" checkpoints to keep concurrently, ie. saved with `step` and not `save`. `None` for keep all
        """
        if self == CheckpointRetentionPolicy.ALL:
            return None
        return 1



def _is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False



def find_all_checkpoints(ckpt_dir: str | Path) -> List[Path]:
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return []
    
    checkpoints = [p for p in ckpt_dir.iterdir() if p.is_dir() and _is_int(p.name)]
    checkpoints.sort(key=lambda p: int(p.name))
    return checkpoints


def find_latest_checkpoint(ckpt_dir: Path | str) -> Path | None:
    checkpoints = find_all_checkpoints(ckpt_dir)
    if len(checkpoints) == 0:
        return None
    return checkpoints[-1]


def keep_last_n_checkpoints(ckpt_dir: Path | str, n: int | None):
    if n is None:
        return 
    
    checkpoints = find_all_checkpoints(ckpt_dir)
    if ckpt_dir in checkpoints[:-n]:
        try:
            shutil.rmtree(ckpt_dir)
            logger.info(f"Deleted: {ckpt_dir}")
        except Exception:
            logger.exception(f"failed to delete: {ckpt_dir}")
        

def keep_checkpoint_copy(src: Path | str):
    src = Path(src)
    dst = src.parent / f"{src.name}_keep"
    subprocess.check_output(["cp", "--recursive", "--link", src, dst])
    logger.info(f"copied: {src} -> {dst}")



def cleanup_checkpoint(
    ckpt_dir: str, 
    checkpoint_retention_policy: CheckpointRetentionPolicy
):
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return []
    
    checkpoint_filters = checkpoint_retention_policy.keep_filters
    checkpoints = [p for p in ckpt_dir.iterdir() if p.is_dir()]

    for checkpoint in checkpoints:
        if checkpoint in checkpoint_filters:
            continue
        try:
            shutil.rmtree(checkpoint)
            logger.info(f"Deleted: {checkpoint}")
        except Exception:
            logger.exception(f"Failed to delete :{checkpoint}")
            

def save_checkpoint(
    ckpt_dir: str | Path,
    *,
    iteration: int | str,
    params, # : pytree
    optimizer_state, # : pytree
    overwrite: bool = True,
    **others
):
    ckpt_dir = Path(ckpt_dir).absolute()
    ckpt_dir_exists = ckpt_dir.exists()

    if ckpt_dir_exists:
        if overwrite:
            if ckpt_dir.is_dir():
                shutil.rmtree(ckpt_dir)
            else:
                ckpt_dir.unlink()
            logger.info(f"Deleted: {ckpt_dir}")
        else:
            raise RuntimeError(f"Checkpoint already exists: {ckpt_dir}")

    state_to_save = {
        "iteration": iteration
    }
    state_to_save["model_params"] = params
    if optimizer_state is not None:
        state_to_save["optimizer_state"] = optimizer_state

    state_to_save.update(others)
    ocp.PyTreeCheckpointer().save(ckpt_dir, state_to_save)
    logger.info(f"Saved: {ckpt_dir}")



def load_checkpoint(
        ckpt_dir: str | Path,
        *,
        abstract_model_params, # pytree
        abstract_optimizer_state, # pytree
        strict_loading: bool = True,
        **others
) -> dict | None: # pytree
    ckpt_dir = Path(ckpt_dir).absolute()
    state_to_load = {
        "iteration": int,
        "model_params": abstract_model_params
    }

    if abstract_optimizer_state is not None:
        state_to_load["optimizer_state"] = abstract_optimizer_state

    state_to_load.update(others)


    checkpoint = ocp.PyTreeCheckpointer().restore(
        ckpt_dir, 
        args=ocp.args.PyTreeRestore(
            item=state_to_load,
            partial_restore=not strict_loading
        )
    )
    return checkpoint


"""
(AFAIK rn):
    register_dont_save_hooks # won't be needed
    init_fsdp_model_from_checkpoint # not in use 
    init_model_from_checkpoint_for_evals # not in use
"""
