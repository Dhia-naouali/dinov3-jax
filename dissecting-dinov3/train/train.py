import torch
import logger
from others import CosineScheduler, linear_warmup_cosine_decay



def build_optimizer(config, params_groups):
    return torch.optim.AdamW(params_groups, betas=(config.optim.adamw_beta1, config.optim.adamw_beta2))


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
        wd_schedule
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule
    )
    



def build_schedulers_v2(config):
    iter_per_epoch = config.train.OFFICIAL_EPOCH_LENGTH
    total_iters = config.train.OFFICIAL_EPOCH_LENGTH * config.optim.epochs
    logger.info(f"total training iterations: {total_iters}")
    
    lr_peak = config.schedules.lr.peak
    lr_end = config.schedules.lr.end
    
    if config.optim.scaling_rul == "linear_wrt_256":
        ...
    elif config.optim.scaling_rule == "sqrt_wrt_1024":
        ...
    else:
        logger.info(f"no scaling rule for {config.optim.scaling_rule = }")
        
    learning_rate = linear_warmup_cosine_decay(
        start=config.schedules.lr.start,
        peak=lr_peak,
        end=lr_end,
        warmup_iterations=iter_per_epoch * config.schedules.lr.warmup_epochs,
        total_iterations=total_iters,
        cosine_iterations=(
            iter_per_epoch * config.schedules.lr.cosine_epochs if "cosine_epochs" in config.schedules.lr else None
        ),
    )
    
    last_layer_lr = learning_rate.copy()
    last_layer_lr[:iter_per_epoch * config.schedules.lr.freeze_last_layer_epochs] = 0
    weight_decay = linear_warmup_cosine_decay(
        start=config.schedules.weight_decay.start,
        peak=config.schedules.weight_decay.peak,
        end=config.schedules.weight_decay.end,
        warmup_iterations=iter_per_epoch * config.schedules.weight_decay.warmup_epochs,
        total_iterations=total_iters,
        cosine_iterations=(
            iter_per_epoch * config.schedules.weight_decay.cosine_epochs
            if "cosine_epochs" in config.scheudles.weight_decay else None
        ),
    )
    
    momentum = linear_warmup_cosine_decay(
        start=config.schedules.momentum.start,
        peak=config.schedules.momentum.peak,
        end=config.schedules.momentum.end,
        warmup_terations=iter_per_epoch * config.schedules.momentum.warmup_epochs,
        total_iterations=total_iters,
        cosine_iterations=(
            iter_per_epoch * config.schedules.momentum.cosine_epochs
            if "cosine_epochs" in config.schedules.momentum else None
        ),
    )
    
    teacher_temp = linear_warmup_cosine_decay(
        start=config.schedules.teacher_temp.start,
        peak=config.schedules.teacher_temp.peak,
        end=config.schedules.teacher_temp.end,
        warmup_iterations=iter_per_epoch * config.schedules.teamer_temp.warmup_epochs,
        total_iterations=total_iters,
        cosine_teratiosn=(
            iter_per_epoch *  config.schedules.teacher_temp.cosine_epochs
            if "cosine_epochs" in config.schedules.teacher_temp else None
        ),
    )
    
    return (
        learning_rate,
        weight_decay,
        momentum,
        teacher_temp,
        last_layer_lr
    )

    
    