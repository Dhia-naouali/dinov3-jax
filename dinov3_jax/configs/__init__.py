# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3

# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.


from .config import (
    DinoV3SetupArgs,
    apply_scaling_rules_to_cfg,
    exit_job,
    get_cfg_from_args,
    get_default_config,
    setup_config,
    setup_job,
    setup_multidistillation,
    write_config,
)