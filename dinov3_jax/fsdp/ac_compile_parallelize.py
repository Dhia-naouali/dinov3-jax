# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3

# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.


import logging
import jax
import logging
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P


logger = logging.getLogger("dinov3")



def ac_compile_parallelize(trained_model, inference_only_models, config, min_shard_size=2**12):
    axis_name = "dp"
    axis_size = jax.device_count()

    mesh = jax.make_mesh((jax.device_count(),), (axis_name,))


    def _find_shard_axis(p):
        idx = np.argsort(p.shape)[::-1]
        partition_axes = [None] * len(p.shape)
        for i in idx:
            if p.shape[i] % axis_size == 0:
                partition_axes[i] = axis_name
                return tuple(partition_axes)
        logging.info("Parameter {p.shae} can't be sharded on any axis")
        return tuple(partition_axes)

    def _shard_param(p):        
        if p.ndim == 1:
            return p
        
        p_shape = _find_shard_axis(p)
        print(p.shape, p_shape)
        return jax.device_put(p, NamedSharding(mesh, P(*p_shape)))
    
    return jax.tree_util.tree_map(_shard_param, trained_model)