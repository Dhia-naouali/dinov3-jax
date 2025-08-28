import logging
import argparse
import numpy as np


import optax

from cosine_lr_scheduler import CosineScheduler

logger = logging.getLogger("dinov3")







def get_parsed_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true", help="eval only")
    parser.add_argument("--eval", type=str, default="", help="eval type")
    parser.add_argument("--eval-pretrained-weights", type=str, default="", help="path to weights")
    
    # to inspect
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER)
    
    parser.add_argument("--output-dir", type=str, default="./local_dino")
    parser.add_argument("seed", default=12, type=int, help="rng seed")
    parser.add_argument("--test-ibot", action="store_true", help="test ibot")
    parser.add_argument("--profiling", action="store_true", help="profile")
    parser.add_argument("--dump-fsdp-weights", action="store_true", help="save sharded")
    parser.add_argument("--record-ref-losses", action="store_true", help="reference losses")
    parser.add_argument("--ref-losses-path", default="", type=str)
    parser.add_argument("--multi-distillation", action="store_true")
    
    return parser.parse_args()


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
        total_iters=config.optim["epochs"] * OFFICIAL_EPOCCH_LENGTH
        warmup_iters=config.optim["warmup_epochs"] * OFFICIAL_EPOCCH_LENGTH
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