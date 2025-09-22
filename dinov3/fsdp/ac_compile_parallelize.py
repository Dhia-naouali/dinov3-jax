import logging
from functools import partial
from typing import Any, List, Optional


# import torch 
# import torch.distributed as dist
# from torch import nn
# from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
# from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
# from torch.distributed.fsdp import register_fsdp_forward_method
# from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState
# from torch.utils.checkponit import create_selective_checkpoint_contexts

from dinov3.utils import utils

logger = logging.getLogger("dinov3")

def ac_compile_parallelize(
        trained_model,
        inference_only_models,
        config: Any,
):
    """
    this implementation significantly deviate from the original repo / pytorch implementation
    due to the design choices / architecture differences between JAX and PyTorch
    the function name is kept the same to avoid evitable changes to the repo structure during translation
    """

    # 1. AC: mainly unnecessary since XLA has a global graph awareness and does already optimize compute / save tradeoffs
    if config.train.checkpointing_full:
        print("block full checkpoint (to implement)")
    else:
        logger.info(
            "manual selective checkpointing ignored  have faith-in-XLA"
            "if this melted your GPU please open an issue (at github.com/openxla/xla)"
        )
    
    import IPython; IPython.embed()
    # 2. compile: will be performed using func transform / decorator



    # 3. FSDP
    


