# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3

# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.


import logging
from pathlib import Path

from . import vision_transformer as vits

logger = logging.getLogger("dinov3")


def build_model(args, only_teacher=False, img_size=224):
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            pos_embed_rope_base=args.pos_embed_rope_base,
            pos_embed_rope_min_period=args.pos_embed_rope_min_period,
            pos_embed_rope_max_period=args.pos_embed_rope_max_period,
            pos_embed_rope_normalize_coords=args.pos_embed_rope_normalize_coords,
            pos_embed_rope_shift_coords=args.pos_embed_rope_shift_coords,
            pos_embed_rope_jitter_coords=args.pos_embed_rope_jitter_coords,
            pos_embed_rope_rescale_coords=args.pos_embed_rope_rescale_coords,
            qkv_bias=args.qkv_bias,
            layerscale_init=args.layerscale,
            norm_layer=args.norm_layer,
            ffn_layer=args.ffn_layer,
            ffn_bias=args.ffn_bias,
            proj_bias=args.proj_bias,
            n_storage_tokens=args.n_storage_tokens,
            mask_k_bias=args.mask_k_bias,
            untie_cls_and_patch_norms=args.untie_cls_and_patch_norms,
            untie_global_and_local_cls_norm=args.untie_global_and_local_cls_norm,
        )
        
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        # teacher = init_fp8(teacher, args)
        if only_teacher:
            return teacher, teacher.embed_dim
        
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
        )
        embed_dim = student.embed_dim
    else:
        raise NotImplementedError(f"unrecognized architecture {args.arch}")
    # student = init_fp8(student, args)
    
    return student, teacher, embed_dim


def build_model_from_cfg(config, only_teacher=False):
    outputs = build_model(
        config.student,
        only_teacher=only_teacher,
        img_size=config.crops.global_crops_size
        if isinstance(config.crops.global_crops_size, int)
        else max(config.crops.global_crops_size)
    )
    
    if only_teacher:
        teacher, embed_dim = outputs
        return teacher, embed_dim
    else:
        student, teacher, embed_dim = outputs
        return student, teacher, embed_dim


def build_model_for_eval(config, pretrained_weights, shard_unsharded_model=False):
    model, _ = build_model_from_cfg(config, only_teacher=True)
    if pretrained_weights is None or pretrained_weights == "":
        logger.info("No pretrained weights")
    elif Path(pretrained_weights).is_dir():
        logger.info("DCP (?) checkpoint")
        from dinov3.checkpointer import load_checkpoint
        from dinov3.fsdp.ac_compile_parallelize import ac_compile_parallelize
        
        moduledict = {"backbone": model}
        ac_compile_parallelize(moduledict, inference_only_models=[], config=config)
        
        load_checkpoint(pretrained_weights, model=moduledict, strict_loading=True)
        shard_unsharded_model = False
    else:
        logger.info("consolidated (?) checkpoint")
        from dinov3.checkpointer import init_model_from_checkpoint_for_evals
        
        init_model_from_checkpoint_for_evals(model, pretrained_weights, "teacher")
    
    if shard_unsharded_model:
        logger.info("sharding model")
        moduledict = {"backbone": model}
        ac_compile_parallelize(moduledict, inference_only_models=[], config=config)
    return model