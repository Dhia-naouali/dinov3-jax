# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3

# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.


# this file was imported as is from the original repo since no changes are needed to adapt to flax/jax

import contextlib
from datetime import timedelta
from typing import Optional

from dinov3_jax.configs import exit_job, setup_job


@contextlib.contextmanager
def job_context(
    output_dir: Optional[str] = None,
    distributed_enabled: bool = True,
    logging_enabled: bool = True,
    seed: Optional[int] = 0,
    restrict_print_to_main_process: bool = True,
    distributed_timeout: timedelta | None = None,
):
    setup_job(
        output_dir=output_dir,
        distributed_enabled=distributed_enabled,
        logging_enabled=logging_enabled,
        seed=seed,
        restrict_print_to_main_process=restrict_print_to_main_process,
        distributed_timeout=distributed_timeout,
    )
    try:
        yield
    finally:
        exit_job(distributed_enabled=distributed_enabled, logging_enabled=logging_enabled)