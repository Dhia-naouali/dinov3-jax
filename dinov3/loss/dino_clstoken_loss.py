# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3

import jax 
import jax.numpy as jnp
import flax.linen as nn


class DINOLoss(nn.Module):
    out_dim: int
    student_temp: float = .1
    center_momentum: float = .9
    
    def setup(self):
        self.center = self.variable(
            "state", "center", lambda: jnp.zeros((1, self.out_dim))
        )

    def softmax_center_teacher(
        self, teacher_output, teacher_temp, update_centers=True
    ):
        if update_centers:
            self.apply_center_update(teacher_output)
        return nn.softmax(
            (teacher_output - self.center.value) / teacher_temp, 
            axis=-1
        )


    def sinkhorn_knopp_teacher(
        self, teacher_output, teacher_temp, n_iterations=3, init_phase=False
    ):
        world_size = jax.device_count()
        Q = jnp.exp(teacher_output / teacher_temp).T
        B = Q.shape[1] * world_size
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



    def __call__(self, student_logits, teacher_probs, ignore_diagonal=False):
        S, B, _ = student_logits.shape
        T, _, _ = teacher_probs.shape
        student_probs = nn.log_softmax(student_logits / self.student_temp, axis=-1)

        def ignore_diagonal_fn(operands):
            student_probs, teacher_probs, B, S, T = operands
            loss = -jnp.einsum("sbk, tbk -> st", student_probs, teacher_probs)

            loss = jnp.fill_diagonal(loss, 0., inplace=False)
            M = jnp.minimum(S, T)
            return loss.sum() / (B * S * T - B * M)

        def no_ignore_diagonal_fn(operands):
            student_probs, teacher_probs, B, S, T = operands
            loss = -jnp.einsum("sbk, tbk -> ", student_probs, teacher_probs)
            return loss / (B * S * T)
        
        return jax.lax.cond(
            ignore_diagonal,
            ignore_diagonal_fn,
            no_ignore_diagonal_fn,
            operand=(student_probs, teacher_probs, B, S, T)
        )

    def apply_center_update(self, teacher_output):
        local_center = jnp.mean(teacher_output, axis=0, keepdims=True)
        global_center = jax.lax.pmean(local_center, axis_name="dp")
        self.center.value = self.center.value * self.center_momentum +\
            global_center * (1 - self.center_momentum)