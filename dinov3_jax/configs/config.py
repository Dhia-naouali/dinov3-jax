# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3

import os
import sys
import math
import logging
import pathlib
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, List, Optional, Sequence, Tuple

from omegaconf import DictConfig, OmegaConf

import dinov3_jax.distributed as distributed
from dinov3_jax.logging import cleanup_logging, setup_logging
# from dinov3_jax.logging import setup_logging
# from dinov3_jax.utils import fix_random_seeds, get_conda_env, get_sha
from dinov3_jax.utils import fix_random_seeds


logger = logging.getLogger("dinov3")

@dataclass
class DinoV3SetupArgs:
    config_file: str
    pretrained_weights: str | None = None
    shard_unsharded_model: bool = False
    output_dir: str = ""
    opts: List[Any] = field(default_factory=lambda: [])
    
    def __post_init__(self):
        
        if OmegaConf.is_config(self.opts):
            self.opts = OmegaConf.to_object(self.opts)
    

def apply_scaling_rules_to_cfg(config):
    # assert distributed.is_enabled(), "setup distributed to get global size !"
    if "schedules" in config:
        return config
    
    if config.optim.scaling_rule == "linear_wrt_256":
        old_lr = config.optim.lr
        config.optim.lr *= config.train.batch_size_per_gpu * distributed.get_world_size() / 256.
        logger.info(f"linear scaling learning rate; old: {old_lr}, new: {config.optim.lr}")
    elif config.optim.scaling_rule == "sqrt_wrt_1024":
        old_lr = config.optim.lr
        config.optim.lr *= 4 * math.sqrt(config.train.batch_size_per_gpu * distributed.get_world_size() / 1024.)
        logger.info(f"sqrt scaling learning rate; old: {old_lr}, new: {config.optim.lr}")
    return config

def write_config(config, output_dir, name="config.yaml"):
    logger.info(OmegaConf.to_yaml(config))
    output_dir = os.path.abspath(output_dir)
    saved_config_path = os.path.join(output_dir, name)
    with open(saved_config_path, "w") as f:
        OmegaConf.save(config=config, f=f)
    return saved_config_path


def get_default_config():
    p = pathlib.Path(__file__).parent / "ssl_default_config.yaml"
    return OmegaConf.load(p)

def get_cfg_from_args(args: DinoV3SetupArgs, multidistillation=False, strict=True):
    overrides = [*args.opts]
    if args.output_dir is not None:
        overrides.append(f"train.output_dir={os.path.realpath(args.output_dir)}")
        
    config = OmegaConf.load(args.config_file)
    opts_config = OmegaConf.from_cli(overrides)
    
    if multidistillation:
        config = OmegaConf.merge(config, opts_config)
    else:
        default_config = get_default_config()
        if strict:
            OmegaConf.set_struct(default_config, True)
        config = OmegaConf.merge(default_config, config, opts_config)
    return config

def setup_config(args: DinoV3SetupArgs, strict_cfg=True):
    config = get_cfg_from_args(args, strict=strict_cfg)
    logger.info("\n".join(
        f"{k}: {str(v)}" for k, v in sorted(dict(vars(args)).items())
    ))
    
    if args.output_dir is not None:
        write_config(config, args.output_dir)

    
    apply_scaling_rules_to_cfg(config)
    return config

# def _enumerate_all_subgroup_ranks(all_subgroup_rank_spans: Sequence[Tuple[int, int]]):
#     ...

def setup_multidistillation(args: DinoV3SetupArgs):
    ...




def setup_job(
        output_dir=None,
        distributed_enabled=True,
        logging_enabled=True,
        seed=12,
        restricted_print_to_main_process=True,
        distributed_timeout=None
):
    if output_dir is not None:
        output_dir = os.path.relpath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    if logging_enabled:
        setup_logging(
            output=output_dir,
            level=logging.INFO,
            log_to_stdout_only_in_main_process=restricted_print_to_main_process,
        )
    
    if distributed_enabled:
        print("distributed thingy not setup atp")
        # distributed.enable(
        #     overwrite=True,

        # )

    if seed is not None:
        # rank = distributed.get_rank()
        # fix_random_seeds(seed + rank)
        fix_random_seeds(seed) # temp
    
    # logger = logging.getLogger("dinov3")
    # logger.info(f"git:\n    {get_sha()}\n")

    # conda thingies ...
    


def exit_job(distributed_enabled=True, logging_enabled=True):
    ...