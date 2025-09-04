import jax.numpy as jnp
import flax.linen as nn


class GramLoss(nn.Module):
    apply_norm: bool = True
    img_level: bool = True
    remove_neg: bool = True
    remove_only_teacher_neg: bool = False

    def setup(self):
        assert self.remove_neg != self.remove_only_teacher_neg


    def __call__(self, output_feats, target_feats, img_level=True):
        # image level gram matrix: B, N, D
        if img_level:
            assert len(target_feats.shape) == 3 and len(output_feats.shape) == 3
        
        if self.apply_norm:
            target_feats /= jnp.linalg.norm(target_feats, axis=-1, keepdims=True)
            output_feats /= jnp.linalg.norm(output_feats, axis=-1, keepdims=True)
        
        # batch level gram matrix: B*N, D
        if not img_level:
            if len(target_feats.shape) == 3:
                target_feats = target_feats.reshape(-1, target_feats.shape[-1])
            
            if len(output_feats.shape) == 3:
                output_feats = output_feats.reshape(-1, output_feats.shape[-1])

        target_sim = target_feats @ jnp.moveaxis(target_feats, -1, -2)
        student_sim = output_feats @ jnp.moveaxis(output_feats, -1, -2)

        if self.remove_neg:
            target_sim = jnp.where(target_sim < 0., 0., target_sim)
            student_sim = jnp.where(student_sim < 0., 0., student_sim)
        
        elif self.remove_only_teacher_neg:
            target_sim = jnp.where(target_sim < 0., 0., target_sim)
        
        return jnp.mean((student_sim - target_sim) ** 2)