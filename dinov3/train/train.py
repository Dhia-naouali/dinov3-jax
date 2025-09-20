# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3

import gc
import os
import sys
import copy
import math
import logging
import argparse
from pathlib import Path
from functools import partial
import torch

import jax
import optax
import jax.numpy as jnp

from dinov3.train.cosine_lr_scheduler import CosineScheduler, linear_warmup_cosine_decay
from dinov3.train.ssl_meta_arch import SSLMetaArch
from dinov3.train.multidist_meta_arch import MultiDistillationMetaArch
from dinov3.configs import setup_job, setup_config
from dinov3.logging import setup_logging, MetricLogger
from dinov3.data import MaskingGenerator, make_dataset, make_data_loader, collate_data_and_cast, SamplerType

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


def build_optimizer(config, params_groups):
    for params_group in params_groups:
        # extract schedule
        # init optimizer
    
    # return optimizers




    # return None
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
        trunc_extra=config.optim["schedule_trunc_extra"],
    )
    
    weight_decay = dict(
        base_value=config.optim["weight_decay"],
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

    logger.info(f"Making meta arch {meta_arch.__name__}")
    fake_batch = {
        "collated_global_crops": jnp.ones((2 * 4, 224, 224, 3)),  # (2 global crops per image * batch_size=4)
        "collated_local_crops": jnp.ones((8 * 4, 96, 96, 3)),     # (local crops)
        "collated_masks": jnp.ones((8, 196)),                     # fake patch masks
        "mask_indices_list": jnp.arange(68*4),                    # indices
        "masks_weight": jnp.ones((8,)),                           # weights
        "n_masked_patches": jnp.array([50, 60, 70, 80]),
        "upperbound": 1.0,
        "global_batch_size": 4,
    }
    key = jax.random.key(1)
    model = meta_arch(config)
    # fill with nans to check for init
    logger.info(f"Model after distributed #### TO FIX ####:\n{model}")
    init_params = model.init(key, fake_batch, teacher_temp=.7, iteration=0)
    # main_key = jax.random.PRNGKey(12)
    # main_key, init_key = jax.random.split(main_key)
    # input_shape = ...
    
    # params = model.init(init_key, jnp.zeros(input_shape))
    
    # prepare for FSDP (replicate across devices ?)
    logger.info(f"...") # jax.debug.visualize_array_sharding ???
    print(args.eval_only)
    if args.eval_only:
        iteration = model.get_checkpointer_class()(
            model, save_dir=config.train.output_dir
        ).resume_or_load(
            config.MODEL.WEIGHTS, resume = not args.no_resume
        ).get("teration", 1) + 1
        
        return do_test(config, model, f"manual_{iteration}")
    do_train(config, (model, init_params), resume=not args.no_resume)



def do_train(config, model_n_params, resume=False):
    model, init_params = model_n_params
    # no process subgroups
    ckpt_dir = Path(config.train.output_dir, "ckpt").expanduser()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    param_groups = model.get_params_groups(init_params["params"])
    optimizer = build_optimizer(config, param_groups)

    optimizer_state = optimizer.init(init_params)
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule
    ) = build_schedulers(config)


    if config.multidistillation.enabled:
        register_dont_save_hooks(
            model,
            dont_save=["teacher params"]
        )

    start_iter = 0
    if resume: # and (last_checkpoint_dir := find_latest_checkpoint(ckpt_dir)):
        ...
        # raise Exception("resume zeft")
        # logger.inof(f"checkpoint found {last_checkpoint_dir}")
        # start_iter = (
        #     load_checkpoint(
        #         last_checkpoint_dir,
        #         model=model,
        #         optimizer=optimizer,
        #         strict_loading=False,
        #         # no process groups
        #     )
        # ) + 1

    OFFICIAL_EPOCH_LENGTH = config.train.OFFICIAL_EPOCH_LENGTH
    max_iter = config.optim.epochs * OFFICIAL_EPOCH_LENGTH

    # if config.multidistillation.enabled:
    #     global_batch_size = config.multidistillation.global_batch_size
    # else:
    #     # * GPUs per host * num_hosts
    #     global_batch_size = config.train.batch_size_per_gpu * distributed.get_world_size()
    
    global_batch_size = config.train.batch_size_per_gpu * jax.device_count()

    data_loader = build_multi_resolution_data_loader_from_cfg(
        config=config,
        model=model,
        start_iter=start_iter,
    )


    logger.info(f"Starting training from iteration {start_iter}")
    metrics_file = os.path.join(config.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    # gc.disable()
    gc.collect()

    next(iter(data_loader))

    # student = model.student
    iteration = start_iter
    num_gram_updates = 0
    if (
        config.gram.use_loss
        and model.has_gram_teacher
        and config.gram.rep_update
        and start_iter > 0
        and start_iter >= config.gram.it_first_update
    ):
        num_gram_updates = math.ceil((start_iter + 1 - config.gram.it_first_update)  / config.gram.update_frequency)
        logger.info(f"Gram was updated {num_gram_updates} times before iteration {start_iter}")


    consecutive_nan_count = 0

    for data in metric_logger.log_every(
        data_loader, print_freq=10,
        header="Training",
        n_iterations=max_iter,
        start_iteration=start_iter
    ):
        it = iteration
        data["global_batch_size"] = global_batch_size
        if iteration > max_iter:
            return
        
        if (iteration + 1) % 150 == 0:
            logger.info("Gargage collection")
            gc.collect()
        
        if config.gram.use_loss and model.gram_it_load_ema_teacher == it:
            logger.info(f"Loading EMA teacher info Gram teacher before iteration {it}")
            model.gram_load_ema_teacher() # not implemented yet
        
        lr = lr_schedule[it]
        wd = wd_schedule[it]
        mom = momentum_schedule[it]
        teacher_temp = teacher_temp_schedule[it]
        last_layer_lr = last_layer_lr_schedule[it]
        train_loss, metrics_dict = model.apply(init_params, data, teacher_temp=teacher_temp, iteration=it, rngs={
            "dropout": jax.random.PRNGKey(1), "drop_path": jax.random.PRNGKey(2)
        })


        if config.optim.clip_grad:
            print("to clip grads")



        # reduce loss & metric logs
        total_loss_all_ranks = ...


        if total_loss_all_ranks.isnan().any():
            ...
        else:
            consecutive_nan_count = 0


        optimizer.apply(...)
        model.update_ema(mom)

        import IPython; IPython.embed()

        # apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)
        # zero grad

        # total_loss, metrics_dict = model.forward_backward(data, teacher_temp, iteration=it)

        if config.optim.clip_grad:
            ...


    
    
    OFFICIAL_EPOCH_LENGTH = config.train.OFFICIAL_EPOCH_LENGTH
    max_iter = config.optim.epoch * OFFICIAL_EPOCH_LENGTH
    if config.multidistillation.enabled:
        raise




def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    ...



def build_multi_resolution_data_loader_from_cfg(config, model, start_iter, seed=65537):
    global_crops_sizes = (
        [config.crops.global_crops_size] 
        if isinstance(config.crops.global_crops_size, int) 
        else config.crops.global_crops_size
    )
    local_crops_sizes = (
        [config.crops.local_crops_size]
        if isinstance(config.crops.local_crops_size, int)
        else config.crops.local_crops_size
    )


    gram_teacher_crops_sizes = (
        [config.crops.gram_teacher_crops_size]
        if config.crops.gram_teacher_crops_size is None or isinstance(config.crops.gram_teacher_crops_size, int)
        else config.crops.gram_teacher_crops_size
    )
    loader_ratios = (
        [config.crops.global_local_crop_pairs_ratios]
        if type(config.crops.global_local_crop_pairs_ratios) in [int, float]
        else config.crops.global_local_crop_pairs_ratios
    )
    assert len(global_crops_sizes) == len(local_crops_sizes)  == len(gram_teacher_crops_sizes) == len(loader_ratios)

    loaders = []
    for increment, (
        global_crops_size_i, 
        local_crops_size_i, 
        gram_teacher_crops_size_i
    ) in enumerate(zip(
        global_crops_sizes, local_crops_sizes, gram_teacher_crops_sizes
    )):
        config_i = copy.deepcopy(config)
        config_i.crops.global_crops_size = global_crops_size_i
        config_i.crops.local_crops_size = local_crops_size_i
        config_i.crops.gram_teacher_crops_size = gram_teacher_crops_size_i
        config_i.train.seed = config.train.seed + increment + 1
        loaders.append(build_data_loader_from_cfg(
            config=config_i, model=model, start_iter=start_iter
        ))
    
    if len(loaders) == 1:
        data_loader = loaders[0]
    else:
        data_loader = CombineDataLoader(
            loaders_with_ratios=zip(loaders, loader_ratios),
            batch_size=config.train.batch_size_per_gpu,
            combining_mode=0,
            name="MultiResDL"
        )
    return data_loader



def build_data_loader_from_cfg(
        config,
        model,
        start_iter
):
    img_size = config.crops.global_crops_size
    patch_size = config.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=.5 * img_size // patch_size * img_size // patch_size
    )

    if config.multidistillation.enabled:
        assert config.multidistillation.global_batch_size % 4 == 4, "to fix"
        # ...
        # dataloader_batch_size_per_host = ...
    else:
        local_batch_size = None
        dataloader_batch_size_per_host = config.train.batch_size_per_gpu * jax.local_device_count()


    # to double check
    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=config.ibot.mask_ratio_min_max,
        mask_probability=config.ibot.mask_sample_probability,
        dtype={ # using torch dtypes for data loading, converting to jnp.ndarray later
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16
        }[config.compute_precision.param_dtype],
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        random_circular_shift=config.ibot.mask_random_circular_shift,
        local_batch_size=local_batch_size
    )


    batch_size = dataloader_batch_size_per_host
    num_workers = 0 # config.train.num_workers
    dataset_path = config.train.dataset_path


    dataset = make_dataset(
        dataset_str=dataset_path,
        transform=model.build_data_augmentation_dino(config),
        target_transform=lambda _: (),
    )
    



    if isinstance(dataset, torch.utils.data.IterableDataset):
        sampler_type = SamplerType.INFINITE
    else:
        sampler_type = SamplerType.SHARDED_INFINITE if config.train.cache_dataset else SamplerType.INFINITE
    sampler_type = SamplerType.EPOCH


    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        seed=config.train.seed + start_iter + 1,
        sampler_type=sampler_type,
        sampler_advance=start_iter * dataloader_batch_size_per_host,
        drop_last=True,
        collate_fn=collate_fn,
    )
    return data_loader

    

if __name__ == "__main__":
    main()