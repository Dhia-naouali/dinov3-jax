import jax
import flax.linen as nn
import numpy as np

import logging
from functools import partial
from typing import Callable

logger = logging.getLogger("dinov3")

def shard_params(params, axis_name, min_param_size=2**18):
    axis_idx = jax.lax.axis_index(axis_name)
    axis_size = jax.lax.psum(1, axis_name)

    def _split(x):
        value = x
        names = (None,) * value.ndim

        if isinstance(x, nn.Partitioned):
            logger.warning(f"param {value.shape} already sharded")
            return x
        
        elif value.size <= min_param_size:
            print(f"param {value.shape} isn't sharded (too smol)")
            return x
        
        else:
            shape = value.shape
            idx = np.argsort(shape)[::-1]
            for i in idx:
                if shape[i] % axis_size == 0:
                    split_size = shape[i] // axis_size
                    p_sharded = nn.Partitioned(
                        value=jax.lax.dynamic_slice_in_dim(
                            value, axis_idx * split_size, split_size, axis=i
                        ),
                        names=names[:i] + (axis_name,) + names[i+1:]
                    )
                    return p_sharded
        logger.warning(f"couln't shard param {value.shape} no axis evenly divide the sharding axis")
        return x
    
    return jax.tree_util.tree_map(
        _split, params, is_leaf=lambda x: isinstance(x, nn.Partitioned)
    )


def fwd_gather_bwd_pmean_scatter(x, axis, axis_name):
    axis_size = jax.lax.psum(1, axis_name)

    @jax.custom_gradient
    def f(x):
        def grad_fn(g):
            return (
                jax.lax.psum_scatter(g, axis_name, scatter_dimension=axis, tiled=True) / axis_size
            )
        
        return jax.lax.all_gather(x, axis_name, axis=axis, tiled=True), grad_fn

    return f(x)


def gather_params(params, axis_name):
    def _gather(p):
        if isinstance(p, nn.Partitioned):
            param_shard = p.names
            shard_axis = param_shard.index(axis_name)
            value = fwd_gather_bwd_pmean_scatter(p.value, axis=shard_axis, axis_name=axis_name)

            param_shard = param_shard[:shard_axis] + (None,) + param_shard[shard_axis:]
            return value
        return p
    
    return jax.tree_util.tree_map(
        _gather, params, is_leaf=lambda x: isinstance(x, nn.Partitioned)
    )


def fsdp_wrapper(target: Callable | nn.Module, axis_name, min_param_size=2**4):
    return nn.map_variables(
        target=target,
        trans_in_fn=partial(gather_params, axis_name=axis_name),
        trans_out_fn=partial(shard_params, axis_name=axis_name, min_param_size=min_param_size),
        mapped_collections="params",
        mutable=True
    )





def sync_grads(
    grads,
):
    axis_name="dp"
    def sync_grad(g):
        if isinstance(g, nn.Partitioned):
            return g
        else:
            return jax.lax.pmean(g, axis_name=axis_name)

    return jax.tree_util.tree_map(sync_grad, grads, is_leaf=lambda x: isinstance(x, nn.Partitioned))
