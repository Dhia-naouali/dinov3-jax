# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3

from jax import lax
import jax.numpy as jnp


def reduce_dict(input_dict, average=True):
    reduce = lax.pmean if average else lax.psum
    return {
        k: reduce(v, axis_name="batch") for k, v in input_dict.items()
    }


def _simple_gather_all_tensors(result):
    return lax.all_gather(result, axis_name="batch", axis=0)


def gather_all_tensors(result):
    if result.ndim == 0:
        return _simple_gather_all_tensors(result)
    
    local_size = jnp.array(result.shape)
    local_sizes = lax.all_gather(local_size, axis_name="batch", axis=0)

    max_size = local_sizes.max(axis=0)
    pad_width = [
        (0, m-s) for m, s in zip(max_size, local_size)
    ]
    
    result_padded = jnp.pad(
        result, 
        pad_width
    ) # both jax and pytorch use the same default mode (constant)
    
    gathered_result = lax.gather_all(result_padded, axis_name="batch", axis=0)

    for idx, item_size in enumerate(local_sizes):
        slice_param = [slice(dim_size) for dim_size in item_size]
        gathered_result[idx] = gathered_result[idx][slice_param]
    
    return gathered_result