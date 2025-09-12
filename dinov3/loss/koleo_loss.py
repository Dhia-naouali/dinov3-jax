# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3

from typing import Callable

import jax
import jax.numpy as jnp
import flax.linen as nn


def pairwise_distance(x, y, eps=1e-8):
    return jnp.linalg.norm(x - y, ord=2, axis=-1) + eps


class KoLeoLoss(nn.Module):
    pdist: Callable = pairwise_distance

    def pairwise_NNs_inner(self, x):
        dots = x @ x.T
        dots = dots.at[
            jnp.diag_indices(dots.shape[0])
        ].set(-1.)
        return jnp.argmax(dots, axis=1)

    def __call__(self, student_output, eps=1e-8):
        student_output /= jnp.linalg.norm(student_output, ord=2, axis=-1, keepdims=True) + eps
        indices = self.pairwise_NNs_inner(student_output)
        distances = self.pdist(student_output, student_output[indices])
        loss = -jnp.log(distances + eps).mean()
        return loss



class KoLeoLossDistributed(nn.Module):
    pdist: Callable = pairwise_distance
    topk: int = 1
    loss_group_size: int | None = None

    def setup(self):
        self.topk_fn = jax.vmap(
            lambda x: jax.lax.top_k(x, self.topk)[1]
        )

    def pairwise_NNs_inner(self, x, all_x):
        dots = x @ all_x.T
        local_B, global_B = dots.shape
        device_idx = jax.lax.axis_index("batch") # rank
        dots = dots.at[
            jnp.arange(local_B),
            device_idx * local_B + jnp.arange(local_B)
        ].set(-1.)
        return self.topk_fn(dots)

    def __call__(self, student_output, eps=1e-8):
        student_output /= jnp.linalg.norm(student_output, ord=2, axis=-1) + eps
        all_student_outputs = jax.lax.all_gather(student_output, axis_name="batch").reshape(
            -1, student_output.shape[-1]
        )
        
        indices = self.pairwise_NNs_inner(student_output, all_student_outputs)
        student_expanded = jnp.repeat(student_output, self.topk, axis=0)
        neighbours_output = all_student_outputs[indices.flatten()]
        distances = self.pdist(student_expanded, neighbours_output)
        loss = -jnp.log(distances + eps).mean()
        return loss