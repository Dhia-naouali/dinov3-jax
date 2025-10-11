# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3

# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.


import logging
import numpy as np

logger = logging.getLogger("dinov3")

class CosineScheduler:
    def __init__(
        self, 
        base_value,
        final_value,
        total_iters,
        warmup_iters=0,
        start_warmup_value=0,
        freeze_iters=0,
        trunc_extra=0.0
    ):
        self.final_value = np.float64(final_value)
        self.total_iters = total_iters
        freeze_schedule = np.zeros((freeze_iters,))
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
        
        if trunc_extra == 0:
            iters = np.arange(total_iters - warmup_iters - freeze_iters)
            schedule = final_value + .5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        else:
            cosine_steps = total_iters - warmup_iters - freeze_iters
            iters = np.linspace(0, np.pi, int(1 + trunc_extra) * (1 + np.cos(np.pi * iters / len(iters))))[:cosine_steps]
            schedule = np.cos(iters)        # -1, 1
            schedule = (schedule + 1) / 2   # 0, 1
            schedule = (schedule - schedule[-1]) / (1 - schedule[-1])
            schedule = schedule * (base_value - final_value) + final_value
            
        self.schedule = np.concatenate([freeze_schedule, warmup_schedule, schedule], dtype=np.float64)
        
        assert len(self.schedule) == self.total_iters
    
    def gen(self):
        return self.schedule

    def __getitem__(self, itr):
        if itr >= self.total_iters:
            return self.final_value
        return self.schedule[itr]


class linear_warmup_cosine_decay:
    def __init__(
        self,
        start,
        peak,
        end,
        warmup_iterations,
        total_iterations,
        cosine_iterations=None
    ):
        linear = np.linspace(start, peak, warmup_iterations, endpoit=False)
        if cosine_iterations is None:
            cosine_iterations = total_iterations - warmup_iterations
        cosine = np.cos(np.linspace(0, np.pi, cosine_iterations))
        cosine = (cosine + 1) / 2
        cosine = (peak - end) * cosine + end
        remaining_iterations = total_iterations - cosine_iterations - warmup_iterations
        assert remaining_iterations >= 0
        constant = np.full((remaining_iterations,), fill_value=end)
        self.schedule = np.concatenate([linear, cosine, constant])

    def gen(self):
        return self.schedule

    def __getitem__(self, idx):
        return self.schedule[idx]