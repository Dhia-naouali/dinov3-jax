import math
import jax
import jax.numpy as jnp
import flax.linen as nn



class DINOLoss(nn.Module):
    out_dim: int
    student_temp: float = .1
    center_momentum: float = .9

    def setup(self):
        self.center = self.variable("state", "center", lambda :jnp.zeros((1, self.out_dim)))
        self.updated = True


    def softmax_center_teacher(self, teacher_output, teacher_temp, update_centers=True):
        if update_centers:
            self.apply_center_update()
        return nn.softmax((teacher_output - self.center.value) / teacher_temp, axis=-1)

    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        world_size = jax.device_count()
        dist = world_size > 1
        
        Q = jnp.exp(teacher_output / teacher_temp).T
        B = Q.shape[1] * world_size
        K = Q.shape[0]
        
        sum_Q = jnp.sum(Q)
        if dist: sum_Q = jax.lax.psum(sum_Q) 
        Q /= sum_Q
        
        for _ in range(n_iterations):
            # rows normalization
            sum_of_rows = jnp.sum(Q, axis=1, keepdims=True)
            if dist: sum_of_rows = jax.lax.psum(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # columns normalization
            Q /= jnp.sum(Q, axis=0, keepdims=True)
            Q /= B

        Q *= B
        return Q.T


    def __call__(self, student_logits, teacher_probs, ignore_diagonal=False):
        student_crops, B, K = student_logits.shape
        teacher_crops, _, _ = teacher_probs.shape
        student_logits = nn.log_softmax(student_logits / self.student_temp, axis=-1)

        def ignore_diagonal_fn(student_logits, teacher_probs, B, S, T):
            loss = -jnp.einsum("sbk, tbk -> st", student_logits, teacher_probs)
            M = min(S, T)
            loss = loss.at[
                jnp.diag_indices(M)
            ].set(0.)
            return loss.sum() / (B * S * T - B * M)

        def no_ignore_diagonal_fn(student_logits, teacher_probs, B, S, T):
            loss = -jnp.einsum("sbk, tbk -> ", student_logits, teacher_probs)
            return loss / (B * S * T)
        
        return jax.lax.cond(
            ignore_diagonal,
            ignore_diagonal_fn,
            no_ignore_diagonal_fn,
            operand=(
                student_logits,
                teacher_probs, 
                B,
                student_crops,
                teacher_crops
            )
        )

    def reduce_center_update(self, teacher_output):
        self.updated = False
        self.len_teacher_output = len(teacher_output)
        self.async_batch_center = jnp.sum(teacher_output, axis=0, keepdims=True)
        
        if jax.device_count() > 1: self.async_batch_center = jax.lax.psum(
            self.async_batch_center
        )


    def apply_center_update(self):
        if self.updated is False:
            world_size = jax.device_count()
            
            _t = self.async_batch_center / (self.len_teacher_output * world_size)
            self.center.value = self.center.value * self.center_momentum +\
                _t * (1 - self.center_momentum)
            self.updated = True