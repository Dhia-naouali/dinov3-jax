# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3

# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.


import jax
import random
import jax.numpy as jnp


def cat_keep_shapes(x_list):
    """
    input: list of arrays
    output: 
        - num_tokens, D-1 flattened array
        - original shapes
        - num_tokens
    """
    shapes = [x.shape for x in x_list]
    num_tokens = [x[..., 0].size for x in x_list]
    flattened = jnp.concatenate([x.reshape(-1, x.shape[-1]) for x in x_list])
    return flattened, shapes, num_tokens

def uncat_with_shapes(flattened, shapes, num_tokens):
    """
    cat_keep_shapes^-1
    """
    split_indices = jnp.cumsum(jnp.array(num_tokens))[:-1]
    outputs_splitted = jnp.split(flattened, split_indices, axis=0)
    shapes_adjusted = [shape[:-1] + (flattened.shape[-1],) for shape in shapes]
    outputs_reshaped = [o.reshape(shape) for o, shape in zip(outputs_splitted, shapes_adjusted)]
    return outputs_reshaped


def fix_random_seeds(seed):
    random.seed(seed)


def count_parameters(params):
    c = 0
    import IPython; IPython.embed()
    for value in jax.tree_util.tree_leaves(params):
        c += value.size
    return c


# def named_apply(
#     fn: Callable,
#     module: nn.Module,
#     name: str = "",
#     depth_first: bool = True,
#     include_root: bool = False,
# ) -> nn.Module:
#     if not depth_first and include_root:
#         fn(module=module, name=name)
#     for child_name, child_module in module.named_children():
#         child_name = ".".join((name, child_name)) if name else child_name
#         named_apply(
#             fn=fn,
#             module=child_module,
#             name=child_name,
#             depth_first=depth_first,
#             include_root=True,
#         )
#     if depth_first and include_root:
#         fn(module=module, name=name)
#     return module