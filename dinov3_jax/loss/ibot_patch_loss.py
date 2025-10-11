# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3

import jax
import jax.numpy as jnp
import flax.linen as nn

def lossfunc(t, s, temp):
    return jnp.sum(t * nn.log_softmax(s / temp, axis=-1), axis=-1)



class iBOTPatchLoss(nn.Module):
    patch_out_dim: int
    student_temp: float = .1
    center_momentum: float = .9

    def setup(self):
        self.center = self.variable(
            "state", "center", lambda: jnp.zeros((1, 1, self.patch_out_dim))
        )

    def softmax_center_teacher(
        self, teacher_patch_tokens, teacher_temp, update_centers=True
    ):
        if update_centers:
            self.apply_center_update()
        return nn.softmax(
            (teacher_patch_tokens - self.center.value) / teacher_temp,
            axis=-1
        )

    def __call__(self, student_patch_tokens, teacher_patch_tokens, student_masks_flat):
        t = teacher_patch_tokens # B, N, D
        s = student_patch_tokens # B, N, D
        loss = lossfunc(t, s, self.student_temp)
        loss = jnp.sum(loss * student_masks_flat, axis=-1) / student_masks_flat.sum(axis=-1).clip(1., jnp.inf)
        return - loss.mean()

    def forward_masked(
        self,
        student_patch_tokens_masked,
        teacher_patch_tokens_masked,
        student_masks_flat,
        n_masked_patches: int = None, # to ensure the thingy is jit compilable / won't recompile
        masks_weight=None,
    ):
        t = teacher_patch_tokens_masked
        s = student_patch_tokens_masked
        loss = lossfunc(t, s, self.student_temp)
        if masks_weight is None:
            weights = (
                (1. / student_masks_flat.sum(axis=-1).clip(1., jnp.inf))[:, None]
            )
            masks_weight = jnp.where(
                student_masks_flat, weights, 0.
            )
        if n_masked_patches is not None:
            loss = loss[:n_masked_patches]
        # import IPython; IPython.embed()
        # loss = loss * masks_weight
        return -loss.sum() / student_masks_flat.shape[0]

    def apply_center_update(self, teacher_output):
        local_center = jnp.mean(teacher_output, axis=0, keepdims=True)
        global_center = jax.lax.pmean(local_center, axis_name="dp")
        self.center.value = self.center.value * self.center_momentum +\
            global_center * (1 - self.center_momentum)


    # SinkhornKnoppTeacher
    def sinkhorn_knopp_teacher(
        self, teacher_output, teacher_temp, n_masked_patches_tensor, n_iterations=3, init_phase=False
    ):
        world_size = jax.device_count()
        Q = jnp.exp(teacher_output / teacher_temp).T
        B = n_masked_patches_tensor
        if not init_phase and world_size > 1:
            B = jax.lax.psum(B, axis_name="dp")
        else:
            B = jnp.sum(B)
        K = Q.shape[0]

        sum_Q = jnp.sum(Q)
        if not init_phase and world_size > 1:
            sum_Q = jax.lax.psum(sum_Q, axis_name="dp")

        Q /= sum_Q
        
        for _ in range(n_iterations):
            # rows normalization
            sum_of_rows = jnp.sum(Q, axis=1, keepdims=True)
            if not init_phase and world_size > 1:
                sum_of_rows = jax.lax.psum(sum_of_rows, axis_name="dp")
        
            Q /= sum_of_rows
            Q /= K
            
            # columns normalization
            Q /= jnp.sum(Q, axis=0, keepdims=True)
            Q /= B

        Q *= B
        return Q.T
