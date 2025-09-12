# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3

import os
import sys
import math
import logging
import argparse

import jax
import optax
import jax.numpy as jnp

from dinov3.train.cosine_lr_scheduler import CosineScheduler, linear_warmup_cosine_decay
from dinov3.configs import setup_job, setup_config
# from somewhere import distributed

logger = logging.getLogger("dinov3")



def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true", help="eval only")
    parser.add_argument("--eval", type=str, default="", help="eval type")
    parser.add_argument("--eval-pretrained-weights", type=str, default="", help="path to weights")
    
    # to inspect
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER)
    
    parser.add_argument("--output-dir", type=str, default="./local_dino")
    parser.add_argument("--benchmark-codebase", action="store_true")
    parser.add_argument("seed", default=12, type=int, help="rng seed")
    parser.add_argument("--test-ibot", action="store_true", help="test ibot")
    parser.add_argument("--profiling", action="store_true", help="profile")
    parser.add_argument("--dump-fsdp-weights", action="store_true", help="save sharded")
    parser.add_argument("--record-ref-losses", action="store_true", help="reference losses")
    parser.add_argument("--ref-losses-path", default="", type=str)
    parser.add_argument("--multi-distillation", action="store_true")

    return parser


def build_optimizer(config, schedule):
    return optax.adamw(schedule, b1=config.optim.adamw_beta1, b2=config.optim.adamw_beta2)


def build_schedulers(config):
    if "schedules" in config:
        logger.info("using schedules v2")
        return build_schedulers_v2(config)
    
    OFFICIAL_EPOCCH_LENGTH = config.train.OFFICIAL_EPOCH_LENGTH
    learning_rate = dict(
        base_value=config.optim["lr"],
        final_value=config.optim["min_lr"],
        total_iters=config.optim["epochs"] * OFFICIAL_EPOCCH_LENGTH,
        warmup_iters=config.optim["warmup_epochs"] * OFFICIAL_EPOCCH_LENGTH,
        start_warmup_value=0,
        trunc_extra=config.optim["scedule_trunc_extra"],
    )
    
    weight_decay = dict(
        vase_value=config.optim["weight_decay"],
        final_value=config.optim["weight_decay_end"],
        total_iters=config.optim["epochs"] * OFFICIAL_EPOCCH_LENGTH,
        trunc_extra=config.optim["schedule_trunc_extra"]
    )
    
    momentum = dict(
        base_value=config.teacher["momentum_teacher"],
        final_value=config.teacher["final_momentum_teacher"],
        total_iters=config.optim["epochs"] * OFFICIAL_EPOCCH_LENGTH,
        trunc_extra=config.optim["schedule_trunc_extra"]
    )
    
    teacher_temp = dict(
        base_value=config.teacher["teacher_temp"],
        final_value=config.teacher["teacher_temp"],
        total_iters=config.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCCH_LENGTH,
        warmup_iters=config.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCCH_LENGTH,
        start_warmup_value=config.teacher["warmup_teacher_temp"]
    )
    
    
    lr_schedule = CosineScheduler(**learning_rate)
    wd_schedule = CosineScheduler(**weight_decay)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**learning_rate)
    
    last_layer_lr_schedule.schedule[
        :config.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCCH_LENGTH
    ] = (0)
    
    logger.info("schedulers ready")
    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule
    )





    
def build_schedulers_v2(config):
    iter_per_epoch = config.train.OFFICIAL_EPOCH_LENGTH
    total_iters = config.train.OFFICIAL_EPOCH_LENGTH * config.optim.epochs
    
    lr_peak = config.schedules.lr.peak
    lr_end = config.schedules.lr.end    
    def scale_lr(lr_peak, lr_end, scale):
        logger.info(
            f"scaling rule :{config.optim.scaling_rule}, "
            f"\n\tlr peak: {config.schedules.lr.peak} -> {lr_peak}"
            f"\n\tlr end: {config.schedules.lr.end} -> {lr_end}"
        )
        return lr_peak * scale, lr_end * scale


    if config.optim.scaling_rule == "linear_wrt_256":
        lr_scale = config.train.batch_size_per_device * distributed.get_world_size() / 256.0
        lr_peak, lr_end = scale_lr(lr_peak, lr_end, lr_scale)
    elif config.optim.scaling_rul == "srt_wrt_1024":
        lr_scale = 4 * math.sqrt(config.train.batch_size_per_device * distributed.get_world_size() / 1024.0)
        lr_peak, lr_end = scale_lr(lr_peak, lr_end, lr_scale)
    else:
        logger.info(f"no scaling rule for {config.optim.scaling_rule = }")
    


    learning_rate = dict(
        start=config.schedules.lr.start,
        peak=lr_peak,
        end=lr_end,
        warmup_iterations=iter_per_epoch * config.schedules.lr.warmup_epochs,
        total_iterations=total_iters,
        cosine_iterations=(
            iter_per_epoch * config.schedules.lr.cosine_epochs if "cosine_epochs" in config.schedules.lr else None
        ),
    )

    weight_decay = dict(
        start=config.schedules.weight_decay.start,
        peak=config.schedules.weight_decay.peak,
        end=config.schedules.weight_decay.end,
        warmup_iterations=iter_per_epoch * config.schedules.weight_decay.warmup_epochs,
        total_iterations=total_iters,
        cosine_iterations=(
            iter_per_epoch * config.schedules.weight_decay.cosine_epochs
            if "cosine_epochs" in config.schedules.weight_decay
            else None
        ),
    )

    momentum = dict(
        start=config.schedules.momentum.start,
        peak=config.schedules.momentum.peak,
        end=config.schedules.momentum.end,
        warmup_iterations=iter_per_epoch * config.schedules.momentum.warmup_epochs,
        total_iterations=total_iters,
        cosine_iterations=(
            iter_per_epoch * config.schedules.momentum.cosine_epochs if "cosine_epochs" in config.schedules.momentum else None
        ),
    )
    
    teacher_temp = dict(
        start=config.schedules.teacher_temp.start,
        peak=config.schedules.teacher_temp.peak,
        end=config.schedules.teacher_temp.end,
        warmup_iterations=iter_per_epoch * config.schedules.teacher_temp.warmup_epochs,
        total_iterations=total_iters,
        cosine_iterations=(
            iter_per_epoch * config.schedules.teacher_temp.cosine_epochs
            if "cosine_epochs" in config.schedules.teacher_temp
            else None
        ),
    )

    lr_schedule = linear_warmup_cosine_decay(**learning_rate)
    wd_schedule = linear_warmup_cosine_decay(**weight_decay)
    momentum_schedule = linear_warmup_cosine_decay(**momentum)
    teacher_temp_schedule = linear_warmup_cosine_decay(**teacher_temp)

    last_layer_lr_schedule = linear_warmup_cosine_decay(**learning_rate)
    last_layer_lr_schedule.schedule[: iter_per_epoch * config.schedules.lr.freeze_last_layer_epochs] = 0
    
    return lr_schedule, wd_schedule, momentum_schedule, teacher_temp_schedule, last_layer_lr_schedule




def main(argv=None):
    if argv is None:
        args = get_args_parser().parse_args()
    else:
        args = get_args_parser().parse_args(argv[1:])
        args.output_dir = sys.argv[1]
    if args.multi_distillation:
        print("performing multidistillation run")
        config = setup_multidistillation(args)
        logger.info("setup_multidistillation done")
        assert config.MODEL.META_ARCHITECTURE == "MultiDistillationMetaArch"
    else:
        setup_job(output_dir=args.output_dir, seed=args.seed)
        config = setup_config(args, strict_cfg=False)
        logger.info(config)
        setup_logging(
            output=os.path.join(os.path.abspath(args.output_dir), "nan_logs"),
            name="nan_logger"
        )
    meta_arch = {
        "SSLMetaArch": SSLMetaArch,
        "MultiDistillationMetaArch": MultiDistillationMetaArch
    }.get(config.MODEL.META_ARCHITECTURE, None)
    if meta_arch is None:
        raise ValueError(f"Unkown MODEL.META_ARCHITECTURE {config.MODEL.META_ARCHITECTURE}")


    main_key = jax.random.PRNGKey(config.seed)
    main_key, init_key = jax.random.split(main_key)
    input_shape = ...
    print(config)
    raise Exception()
    
    model = meta_arch(config)
    # fill with nans to check for init
    params = model.init(init_key, jnp.zeros(input_shape))
    
    # prepare for FSDP (replicate across devices ?)
    logger.info(f"...") # jax.debug.visualize_array_sharding ???
    if args.eval_only:
        iteration = model.get_checkpointer_class()(
            model, save_dir=config.train.output_dir
        ).resume_or_load(
            config.MODEL.WEIGHTS, resume = not args.no_resume
        ).get("teration", 1) + 1
        
        return do_test(config, model, f"manual_{iteration}")
    do_train(config, model, resume=not args.no_resume)


if __name__ == "__main__":
    main()