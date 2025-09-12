# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3

import gc
import logging
from functools import partial
from omegaconf import OmegaConf
from typing import Any

import flax.linen as nn
from dinov3.models import build_model_from_cfg
from dinov3.utils import count_parameters
from dinov3.layers.dino_head import DINOHead
from dinov3.loss import DINOLoss, KoLeoLossDistributed, KoLeoLoss, iBOTPatchLoss, GramLoss
from dinov3.train.cosine_lr_scheduler import linear_warmup_cosine_decay
from dinov3.configs import get_default_config

logger = logging.getLogger("dinov3")

class SSLMetaArch(nn.Module):
    config: Any

    def setup(self):
        assert self.config.crops.local_crops_number > 0
        assert self.config.ibot.separate_head is True
        assert self.config.train.centering == "sinkhorn_knopp"

        assert self.config.compute_precision.sharding_strategy == "SHARD_GRAD_OP"
        
        student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(self.config)
        # gc.collect()
        gram_backbone, _ = build_model_from_cfg(self.config, only_teacher=True)
        logger.info(f"Number of parameters: {count_parameters(student_backbone)}")
        
        self.student = {}
        self.teacher = {}
        self.gram_model = {}

        self.student["backbone"] = student_backbone
        self.teacher["backbone"] = teacher_backbone
        self.gram_model["backbone"] = gram_backbone
        logger.inof(f"OPTIONS -- architecture: embed_dim: {embed_dim}")
        self.embed_dim = embed_dim
        self.dino_out_dim = self.config.dino.head_n_prototypes

        logger.info("OPTIONS -- DINO")
        logger.info(f"OPTIONS -- DINO -- loss_weight: {self.config.dino.loss_weight}")
        logger.info(f"OPTIONS -- DINO -- global_ignore_diagonal: {self.config.dino.global_ignore_diagonal}")
        logger.info(f"OPTIONS -- DINO -- head_n_prototypes: {self.config.dino.head_n_prototypes}")
        logger.info(f"OPTIONS -- DINO -- head_bottleneck_dim: {self.config.dino.head_bottleneck_dim}")
        logger.info(f"OPTIONS -- DINO -- head_hidden_dim: {self.config.dino.head_hidden_dim}")
        logger.info(f"OPTIONS -- DINO -- head_norm_last_layer: {self.config.dino.head_norm_last_layer}")
        
        dino_head_class = partial(
            DINOHead,
            in_dim=embed_dim,
            out_dim=self.config.dino_head_n_prototypes,
            hidden_dim=self.config.dino.head_nlayers
        )

        self.student["dino_head"] = dino_head_class()
        self.teacher["dino_head"] = dino_head_class
        self.dino_loss = DINOLoss(self.dino_out_dim)

        logger.info("OPTIONS -- KOLEO")
        logger.info(f"OPTIONS -- KOLEO -- loss_weight: {self.config.dino.koleo_loss_weight}")
        logger.info(f"OPTIONS -- KOLEO -- distributed: {self.config.dino.koleo_loss_distributed}")

        if self.config.dino.koleo_loss_distributed:
            logger.inof(f"OPTIONS -- KOLEO -- topk: {self.config.dino.koleo_topk}")
            logger.inof(f"OPTIOINS -- KOLEO -- distributed_loss_group_size: {self.config.dino.koleo_distributed_loss_group_size}")
            assert self.config.dino.koleo_distributed_replicas == 0, (
                "Option dino.koleo_distributed_replicas is no longer supported"
            )
            self.koleo_loss = KoLeoLossDistributed(
                topk=self.config.dino.koleo_topk,
                loss_group_size=self.config.dino.koleo_distributed_loss_group_size
            )
        else:
            assert self.config.dino.koleo_topk == 1, "Non-distributed Koleo loss only supports 'dino.koleo_topk=1'"
            self.koleo_loss = KoLeoLoss()
        
        logger.info("OPTIONS -- IBOT")
        logger.info(f"OPTIONS -- IBOT -- loss_weight: {self.config.ibot.loss_weight}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_ratio_tuple: {self.config.ibot.mask_ratio_min_max}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_sample_probability: {self.config.ibot.mask_sample_probability}")
        ibot_head_class = partial(
            DINOHead,
            in_dim=self.embed_dim,
            out_dim=self.config.ibot.head_n_prototypes,
            hidden_dim=self.config.ibot.head_hidden_dim,
            bottleneck_dim=self.config.ibot.head_bottleneck_dim,
            nlayers=self.config.ibot.head_nlayers,
        )
        self.student["ibot_head"] = ibot_head_class()
        self.teacher["ibot_head"] = ibot_head_class()
        self.ibot_patch_loss = iBOTPatchLoss(self.config.ibot.head_n_prototypes)
        
        
        self.model_ema = self.teacher # may be overwritten for distillation
        logger.inof(f"Student and Teacher are built: they are both {self.config.student.arch} network")

        if self.config.distillation.enabled:
            self._setup_distillation()
        
        # self.ema_params_lists = None

        self.n_local_crops = self.config.crops.local_crops_number
        self.is_distillation_enabled = self.config.distillation.enabled
        self.dino_global_ignore_diagonal = self.config.dino.global_ignore_diagonal
        self.dino_loss_weight = self.config.dino.loss_weight
        self.dino_koleo_loss_weight = self.config.dino.koleo_loss_weight
        self.ibot_loss_weight = self.config.ibot.loss_weight

        if self.config.dino.reweight_dino_local_loss:
            iter_per_epoch = self.train.OFFICIAL_EPOCH_LENGTH
            total_iterations = iter_per_epoch * self.config.optim.epochs
            schedule_config = self.config.dino.local_loss_weight_schedule
            self.dino_local_loss_schedule = linear_warmup_cosine_decay(
                start=schedule_config.start,
                peak=schedule_config.peak,
                end=schedule_config.end,
                warmup_iterations=iter_per_epoch * schedule_config.warmup_epochs,
                total_iterations=total_iterations,
                cosine_iterations=(
                    iter_per_epoch * schedule_config.cosine_epochs if "cosine_epochs" in schedule_config else None
                )
            )
        
        # gram 
        self.gram_use_loss = self.config.gram.use_loss
        self.gram_ema_teacher = False
        self.has_gram_teacher = False
        self.gram_teacher_initialized = False
        if self.gram_use_loss:
            self.gram_loss = GramLoss(
                apply_norm=self.config.gram.normalized,
                remove_only_teacher_neg=self.config.gram.remove_only_teacher_neg,
                remove_neg=self.config.gram.remove_neg,
            )
            self.has_gram_teacher = True if not self.config.gram.ema_teacher else False
            if self.has_gram_teacher:
                self.gram_teacher = self.gram_model
                logger.info(f"Gram teacher parameter at init: #")
            else:
                self.gram_teacher = self.gram_model = None
            self.gram_loss_weight = self.config.gram.loss_weight
            if self.config.gram.get("loss_weight_schedule"):
                iter_per_epoch = self.config.train.OFFICIAL_EPOCH_LENGTH
                total_iterations = iter_per_epoch * self.config.optim.epochs
                schedule_config = self.config.gram.loss_weight_schedule
                self.gram_loss_schedule = linear_warmup_cosine_decay(
                    start=schedule_config.start,
                    peak=schedule_config.peak,
                    end=schedule_config.end,
                    warmup_iterations=iter_per_epoch * schedule_config.warmup_epochs,
                    total_iterations=total_iterations,
                    cosine_iterations=(
                        iter_per_epoch*schedule_config.cosine_epochs if "cosine_epochs" in schedule_config else None
                    )
                )
                logger.info(f"Applying gram loss weight schedule instead of 'config.gram.loss_weight: {schedule_config}")
            else:
                self.gram_loss_schedule = None
            self.gramçema_teacher = self.config.gram.ema_teacher
            self.gram_ckpt = self.config.gram.ckpt
            self.gram_img_level = self.config.gram.img_level
            self.gram_tokens_used = self.config.gram.tokens_used
            self.gram_rep_update = self.config.gram.rep_update
            self.gram_update_frequency = self.config.gram.update_frequency
            self.gram_it_first_update = self.config.gram.it_first_update
            self.gram_it_load_ema_teacher = self.config.gram.it_load_ema_teacher
            self.gram_compute_stats = self.config.gram.compute_stats
            self.gram_params_lists = None

            if self.gram_ema_teacher and self.gram_ckpt is not None:
                raise ValueError(
                    "Cannot use both 'gram.ema_teacher' and 'gram.ckpt' at the same time, Please set one of them to False"
                )
            if self.gram_ckpt is None and self.gram_it_load_ema_teacher < 0:
                raise ValueError(
                    "if no gram checkpoint is provided, 'gram.it_load_ema_teacher' must be set to a non-negative value"
                )
            
            assert not (self.gram_ema_teacher and self.gram_rep_update)
            assert self.gram_tokens_used in ("all", "masked", "unmasked")
            if self.gram_tokens_used in ["masked", "unmasked"]:
                assert self.gram_img_level is False
            

            logger.info("OPTIONS -- GRAM")
            logger.info(f"OPTIONS -- GRAM -- loss_weight: {self.config.gram.loss_weight}")
            logger.info(f"OPTIONS -- GRAM -- ema teacher: {self.config.gram.ema_teacher}")
            logger.info(f"OPTIONS -- GRAM -- ckpt: {self.config.gram.ckpt}")
            if self.self.config.gram.rep_update:
                logger.info(f"OPTIONS -- GRAM -- repeated update: {self.config.gram.rep_update}")
                logger.info(f"OPTIONS -- GRAM -- update freq: {self.config.gram.update_frequency}")
                logger.info(f"OPTIONS -- GRAM -- iteration first update: {self.config.gram.it_first_update}")

            logger.info(f"OPTIONS -- GRAM -- tokens_used: {self.config.gram.tokens_used}")
            logger.info(f"OPTIONS -- GRAM -- apply normalization: {self.config.gram.normalized}")
            logger.info(f"OPTIONS -- GRAM -- img_level: {self.config.gram.img_level}")
            logger.info(f"OPTIONS -- GRAM -- remove_neg: {self.config.gram.remove_neg}")
            logger.info(f"OPTIONS -- GRAM -- remove_only_teacher_neg: {self.config.gram.remove_only_teacher_neg}")

        if self.config.crops.gram_teacher_crops_size is None and self.has_gram_teacher:
            raise ValueError("config.crops.gram_teacher_crops_size must be set to use gram loss")
        if self.config.crops.gram_teacher_crops_size is not None and self.gram_ema_teacher:
            raise ValueError("config.crops.gram_teacher_crops_size should be None when gram.ema_teacher=True")

        self.student_crop_size = self.config.crops.global_crops_size
        self.gram_global_teacher_resize_method = self.config.gram.global_teacher_resize_methode
        self;gram_global_teacher_resize_antialias = self.config.gram.global_teacher_resize_antialias
        logger.info(f"OPTIONS -- global crops student/teacher size: {self.student_crop_size}")
        logger.info(f"OPTIONS -- global crops GRAM teacher size: {self.config.crops.gram_teacher_crops_size}")
        logger.info(f"OPTIONS -- global crops GRAM teacher resize method: {self.config.gram.global_teacher_resize_method}")
        logger.info(
            f"OPTIONS -- global crops GRAM teacher resize antialias: {self.config.gram.global_teacher_resize_antialias}"
        )


    def _setup_distillation(self):
        logger.info(f"Performing distillation from {self.config.distillation.full_cfg_path}")

        default_config = get_default_config()
        distillation_config = OmegaConf.load(self.config.distillation.full_cfg_path)
        distillation_config = OmegaConf.merge(default_config, distillation_config)

        assert distillation_config.ibot.separate_head is True
        assert distillation_config.ibot.head_n_prototypes == self.config.ibot.head_n_prototypes
        assert distillation_config.dino.head_n_prototypes == self.config.dino.head_n_prototypes
        assert distillation_config.student.patch_size == self.config.student.patch_size

        teacher = {}
        backbone, embed_dim = build_model_from_cfg(distillation_config, only_teacher=True)
        self.teacher["backbone"] = backbone

        self.teacher["dino_head"] = DINOHead(
            in_dim=embed_dim,
            out_dim=distillation_config.dino.head_hidden_dim,
            bottlenech_dim=distillation_config.dino.head_bottleneck_dim,
            nlayers=distillation_config.dino.head_nlayers
        )

        teacher["ibot_head"] = DINOHead(
            in_dim=embed_dim,
            out_dim=distillation_config.ibot.head_n_prototypes,
            hidden_dim=distillation_config.ibot.head_hidden_dim,
            bottleneck_dim=distillation_config.ibot.head_bottleneck_dim,
            nlayers=distillation_config.ibot.head_nlayers
        )
        



    def forward_backward(
            self, data, *, teacher_temp, iteration=0, **ignored_kwargs
    ):
        del ignored_kwargs
        metrics_dict = {}

        n_global_crops = 2
        n_local_crops = self.n_local_crops
        B = data["collated_local_crops"].shape[0] // n_local_crops
        assert data["collated_global_crops"].shape[0] == n_global_crops * B
        metrics_dict["local_batch_size"] = B
        metrics_dict["global_batch_size"] = data["blobal_batch_size"]

        global_crops = data["collated_global_crops"] # to device
        local_crops = data["collated_collated_crops"] # to device
        masks = data["collated_masks"] # to device
        mask_indices_list = data["mask_indices_list"] # to device
        masks_weight = data["masks_weight"] # to device
        n_masked_patches_tensor = data["n_masked_patches"]

        if self.has_gram_teacher:
            assert "collated_gram_teacher_crops" in data, (
                "no gram teacher crops in the data, have you set cfg.crops.gram_teacher_crops_size?"
            )
            gram_teacher_crops = data["collated_gram_teacher_crops"] # to device
        else:
            gram_teacher_crops = None
        
        teacher_global = self.get_teacher_output(
            global_crops.reshape(n_global_crops, B, *global_crops.shape[1:]),
            teacher_temp=teacher_temp,
            n_masked_patches_tensor=n_masked_patches_tensor,
            mask_indices_list=mask_indices_list,
            upperbound=data["upperbound"]
        )

        student_global, student_local = self.get_student_output(
            global_crops=global_crops.reshape(n_global_crops, B, *global_crops.shape[1:]),
            local_crops=local_crops.reshape(n_local_crops, B, *local_crops.shape[1:]),
            upperbound=data["upperbound"],
            masks=masks,
            mask_indices_list=mask_indices_list
        )

        if self.gram_use_loss:
            gram_global = self.get_gram_teacher_output(
                gram_teacher_crops.reshpae(n_global_crops, B, *gram_teacher_crops.shpae[1:]) 
                    if gram_teacher_crops is not None else None,
                masks=masks,
                teacher_global=teacher_global,
                student_global=student_global,
                student_global_crops_size=global_crops.shape[-1]
            )
        else:
            gram_global = {}
        

        loss_accumulator, loss_dict = self.compute_losses(
            teacher_global=teacher_global,
            student_global=student_global,
            student_local=student_local,
            gram_global=gram_global,
            masks=masks,
            mask_indices_list=mask_indices_list,
            masks_weight=masks_weight,
            iteration=iteration,
        )
        
        # back prop loss
        return loss_accumulator, metrics_dict | loss_dict
    

    def get_teacher_output(
            self, images, *, upperbound, mask_indices_list, n_masked_patches_tensor
    ):
        n_crops, B, rgb, H, W = images.shape
        images = images.reshape(-1, rgb, H, W)

        backbone_out = self.teacher.backbone.apply(images, ...)
        cls = backbone_out["x_norm_clstoken"] # n_crops * B, D
        reg = backbone_out["x_storage_tokens"] # n_crops, * B, R, D
        ibot_patch = backbone_out["x_norm_patchtokens"] # n_crops * B, P, D

        # buffer





    def get_student_output(self, *, global_crops, local_crops, upperbound, masks, mask_indices_list):
        n_global_crops, B, rgb, H, W = global_crops.shape
        n_local_crops, B, rgb, H, W = local_crops.shape

        global_crops = global_crops.rehsape(-1, rgb, H, W)

        global_out, local_out = self.student.backbone(
            [global_crops, local_crops.reshape(-1, rgb, H, W)],
            masks=[masks if not self.is_distillabtion_enabled else None, None],
            is_training=True,
        )

        g_cls, g_reg, g_patch = (
            global_out["x_norm_clstoken"],
            global_out["x_storage_tokens"],
            global_out["x_norm_patchtokens"]
        )

        l_cls, l_reg, l_patch = (
            local_out["x_norm_clstoken"],
            local_out["x_storage_tokens"],
            local_out["x_norm_patchtokens"]
        )

        masked_patches_pre_head = ...

        

