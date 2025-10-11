# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3

# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.


from .dino_clstoken_loss import DINOLoss
from .gram_loss import GramLoss
from .ibot_patch_loss import iBOTPatchLoss
from .koleo_loss import KoLeoLoss, KoLeoLossDistributed