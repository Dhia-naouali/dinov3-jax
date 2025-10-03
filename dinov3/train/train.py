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
from typing import Iterable, Any
from functools import partial
from omegaconf import OmegaConf

import torch

import jax
import optax
import jax.numpy as jnp
from flax.training import train_state
import flax.linen as nn
from jax.sharding import PartitionSpec as P


from dinov3.train.cosine_lr_scheduler import CosineScheduler, linear_warmup_cosine_decay
from dinov3.train.ssl_meta_arch import SSLMetaArch
from dinov3.train.multidist_meta_arch import MultiDistillationMetaArch
from dinov3.configs import setup_job, setup_config
from dinov3.logging import setup_logging, MetricLogger
from dinov3.data import MaskingGenerator, make_dataset, make_data_loader, collate_data_and_cast, SamplerType
from dinov3.fsdp.utils import sync_grads
from dinov3.checkpointer import (
    find_latest_checkpoint, 
    keep_checkpoint_copy, 
    keep_last_n_checkpoints, 
    load_checkpoint, 
    save_checkpoint
)

# from somewhere import distributed


# logdir = "/tmp/jax_trace"
# os.makedirs(logdir, exist_ok=True)

logger = logging.getLogger("dinov3")
# jax.config.update('jax_num_cpu_devices', 8)
INIT_PHASE = False


def print_memory_usage(step, total_memory=16 * 1024**3):
    try:
        mem_profile = jax.profiler.device_memory_profile()
        for device in jax.devices():
            mem_info = mem_profile
            used_memory = mem_info
            free_memory = total_memory - used_memory
            print(f"Step {step} - Device {device.id}: Used: {used_memory / 1024**3:.1f} GB, Free: {free_memory / 1024**3:.1f} GB")
    except Exception as e:
        print(f"Step {step}: Failed to get memory profile: {e}")
        for device in jax.devices():
            print(f"Step {step} - Device {device.id}: Memory usage unavailable")







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


def build_optimizer(
        config, 
        param_groups, 
        lr_schedule: Iterable=None, 
        wd_schedule: Iterable=None, 
        last_layer_lr_schedule:Iterable=None
    ):
    if "--groups--" not in param_groups.keys():return None

    # separating the groups configs and param masks + renaming masks to match student params pytree
    groups = param_groups.pop("--groups--")
    for k in list(param_groups.keys()):
        if "student_" not in k:
            param_groups[f"student_{k}"] = param_groups.pop(k)

    lr_schedule = jnp.array(lr_schedule.gen())
    wd_schedule = jnp.array(wd_schedule.gen())
    last_layer_lr_schedule = jnp.array(last_layer_lr_schedule.gen())


    optimizers = optax.multi_transform(
        {
            k: optax.adamw(
                learning_rate=lambda it: v.lr_multiplier * (last_layer_lr_schedule[it] if v.is_last_layer else lr_schedule[it]),
                # weight_decay=lambda it: v.wd_multiplier * wd_schedule[it],
                weight_decay=v.wd_multiplier,
                b1=config.optim.adamw_beta1,
                b2=config.optim.adamw_beta2
            ) for k, v in groups.items()
        },
        param_groups
    )

    """
    current:
        -group1:
            -params
            -schedule1
            -...

    goal:
        dict: 
            params: pytree
            mask1: [...schedules...]
            ...
    """

    return optimizers




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
        logger.info(OmegaConf.to_yaml(config))
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
    model = meta_arch(config)
    # fill with nans to check for init
    logger.info(f"Model after distributed #### TO FIX ####:\n{model}")
    
    # prepare for FSDP (replicate across devices ?)
    # init_params = model.prepare_for_distributed_training(init_params)

    logger.info(f"...") # jax.debug.visualize_array_sharding ???
    if args.eval_only:
        iteration = model.get_checkpointer_class()(
            model, save_dir=config.train.output_dir
        ).resume_or_load(
            config.MODEL.WEIGHTS, resume = not args.no_resume
        ).get("teration", 1) + 1
        
        return do_test(config, model, f"manual_{iteration}")
    do_train(config, model, resume=not args.no_resume)


def do_train(config, model, resume=False):

    # mesh and rngs
    mesh = jax.make_mesh(
        (jax.device_count(),),
        ("dp",)
    )

    main_rng = jax.random.PRNGKey(config.train.seed)
    main_rng, init_rng, dropout_rng, drop_path_rng = jax.random.split(main_rng, 4)


    # no process subgroups
    ckpt_dir = Path(config.train.output_dir, "ckpt").expanduser()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_batch_size = config.train.batch_size_per_gpu * jax.device_count()
    data_loader = build_multi_resolution_data_loader_from_cfg(
        config=config,
        model=model,
        start_iter=0,
    )

    # inits, specs and sharding
    init_batch = next(iter(data_loader))
    init_batch["global_batch_size"] = global_batch_size
    batch_pspec = {
        'collated_global_crops': P("dp"),
        'collated_local_crops': P("dp"),
        "collated_masks": P("dp"),
        "mask_indices_list": P(),
        "n_masked_patches": P(),
        "masks_weight": P(),
        "upperbound": P(),
        "global_batch_size": P(),
    }

    def init_dp(rng, inputs, model):
        init_rng, rng = jax.random.split(rng)
        return model.init(init_rng, inputs, teacher_temp=.7, iteration=0, init_phase=True) # somehow state

    def shard_batch_item(item, spec):
        return jax.device_put(item, jax.sharding.NamedSharding(mesh, spec))

    init_batch = jax.tree_util.tree_map(
        shard_batch_item, 
        init_batch, 
        batch_pspec
    )

    param_specs = nn.get_partition_spec(
        jax.eval_shape(
            jax.shard_map(
                partial(init_dp, model=model),
                mesh=mesh,
                in_specs=(P(), batch_pspec),
                out_specs=P(),
                check_vma=False
            ),
            init_rng,
            init_batch,
        )
    )

    init_fsdp = jax.jit(
        jax.shard_map(
            partial(init_dp, model=model),
            mesh=mesh,
            in_specs=(P(), batch_pspec),
            out_specs=param_specs
        )
    )

    params_fsdp = init_fsdp(init_rng, init_batch)
    ema_params_fsdp = {
        "params": {
            k: jax.tree_util.tree_map(
                lambda x: x.copy(),
                v
            ) for k, v in params_fsdp["params"].items()
            if k.startswith("student_") or k.startswith("teacher_")
        }
    }

    ema_param_specs = {
        "params": {
            k: v for k, v in param_specs["params"].items()
            if k.startswith("student_") or k.startswith("teacher_")
        }
    }
    # import IPython; IPython.embed()

    update_ema = jax.jit(
        jax.shard_map(
            model.update_ema(),
            mesh=mesh,
            in_specs=(ema_param_specs, param_specs, P()),
            out_specs=ema_param_specs
        )
    )


    # schedules, optimizer build & init
    param_groups = model.get_params_groups(params_fsdp["params"])
    student_params = {
        k: v for k, v in params_fsdp["params"].items() if "student_" in k
    }

    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule
    ) = build_schedulers(config)
    optimizer = build_optimizer(
        config, 
        param_groups, 
        lr_schedule=lr_schedule,
        wd_schedule=wd_schedule,
        last_layer_lr_schedule=last_layer_lr_schedule
    )
    student_params = {
        k: v for k, v in params_fsdp["params"].items()
        if "student_" in k
    }

    optimizer_state = optimizer.init(student_params)
    optimizer_specs = nn.get_partition_spec(optimizer_state)



    # the part m undeniably delaying
    if config.multidistillation.enabled:
        register_dont_save_hooks(
            model,
            dont_save=["teacher params"]
        )

    start_iter = 0
    if resume and (last_checkpoint_dir := find_latest_checkpoint(ckpt_dir)):
        logger.inof(f"checkpoint found {last_checkpoint_dir}")
        start_iter = (
            load_checkpoint(
                last_checkpoint_dir,
                model=params_fsdp,
                optimizer=optimizer_state,
                strict_loading=False,
            )
        ) + 1

    OFFICIAL_EPOCH_LENGTH = config.train.OFFICIAL_EPOCH_LENGTH
    max_iter = config.optim.epochs * OFFICIAL_EPOCH_LENGTH

    # if config.multidistillation.enabled:
    #     global_batch_size = config.multidistillation.global_batch_size
    # else:
    #     # * GPUs per host * num_hosts
    #     global_batch_size = config.train.batch_size_per_gpu * distributed.get_world_size()



    logger.info(f"Starting training from iteration {start_iter}")
    metrics_file = os.path.join(config.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    gc.disable()
    gc.collect()

    rngs={"dropout": dropout_rng, "drop_path": drop_path_rng}



    def train_step(
            params,
            batch,
            optimizer_state,
            teacher_temp,
            iteration,
            root_rngs,
            axis_name="dp",
            clip_grads=config.optim.clip_grad
    ):
        student_params = {k: v for k, v in params["params"].items() if "student_" in k}
        axis_idx = jax.lax.axis_index(axis_name)
        rngs = {k: jax.random.fold_in(v, axis_idx) for k, v in root_rngs.items()}
        def loss_fn(student_params):
            temp_params = dict(params)
            temp_params["params"] = {
                k: student_params[k] if "student_" in k else v
                for k, v in temp_params["params"].items()
            }
            loss, metrics_dict = model.apply(temp_params, batch, teacher_temp=teacher_temp, iteration=iteration, rngs=rngs)
            return loss, metrics_dict

        (loss, metrics_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(student_params)

        grads = sync_grads(grads)
        def global_norm(grads):
            return jnp.sqrt(
                sum([
                    jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(grads)
                ])
            )
        

        def clip_grads(grads, max_norm=config.optim.clip_grad):
            norm = global_norm(grads)
            scale = jnp.minimum(1., max_norm / (norm + 1e-6))
            clipped = jax.tree_util.tree_map(
                lambda g: g * scale,
                grads
            )
            return clipped, norm


        if config.optim.clip_grad:
            for k in student_params.keys():
                grads[k], grad_norm = clip_grads(
                    grads[k]
                )
                metrics_dict[f"{k}_grad_norm"] = (
                    grad_norm
                )



        def min_rank_1(v):
            if not isinstance(v, jnp.ndarray):
                v = jnp.array([v])
            
            if v.ndim < 1:
                return jnp.expand_dims(v, 0)
            return v

        metrics_dict = jax.tree_util.tree_map(min_rank_1, metrics_dict)
        metrics_dict = jax.tree_util.tree_map(
            lambda l: jax.lax.pmean(l, axis_name="dp"),
            metrics_dict
        )
    
        # grad norms


        updates, optimizer_state = optimizer.update(grads, optimizer_state, student_params)
        student_params = optax.apply_updates(student_params, updates)
        
        return params, jax.tree_util.tree_map(min_rank_1, optimizer_state), loss, metrics_dict


    metrics_dict_specs = {
        "dino_global_crops_loss": P("dp"),
        "dino_local_crops_loss": P("dp"),

        "dino_local_loss_weight": P(),
        "koleo_loss": P("dp"),

        "local_batch_size": P(),
        "ibot_loss": P("dp"),
    }

    if config.optim.clip_grad:
        metrics_dict_specs = {
            **metrics_dict_specs,
            **{
                f"{k}_grad_norm": P("dp") for k in student_params.keys()
            }
        }


    train_step_fsdp = jax.jit(
        jax.shard_map(
            train_step,
            mesh=mesh,
            in_specs=(
                param_specs,
                batch_pspec,
                optimizer_specs,
                P(), # teacher_temp
                P(), # iteration
                P(), # rngs
            ),
            out_specs=(param_specs, optimizer_specs, P(), P()),
            check_vma=False
        ),
        donate_argnums=(0, 1, 2)
    )


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
        print(iteration)
        if iteration > 256 :# temp, max_iter:
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

        data = jax.tree_util.tree_map(
            shard_batch_item, 
            data, 
            batch_pspec
        )

        # if it % 32 == 0:
        #     try:
        #         jax.profiler.start_trace(f"/tmp/profile-data/step_{it}", create_perfetto_link=True)
        #     except RuntimeError:
        #         jax.profiler.stop_trace()
        #         jax.profiler.start_trace(f"/tmp/profile-data/step_{it}", create_perfetto_link=True)
        
        params_fsdp, optimizer_state, total_loss, metrics_dict = train_step_fsdp(params_fsdp, data, optimizer_state, teacher_temp, it, rngs)

        # if it % 32 == 0:
        #     print(f"iteration {it} mem usage:")
        #     print_memory_usage(it)
        #     jax.profiler.stop_trace()
       
        
        if jnp.isnan(total_loss).any():
            consecutive_nan_count += 1
            # logger.warning("nan loss detected on ranks: unkown") # that's pure rage bating
            logger.warning(f"consecutive NaNs: {consecutive_nan_count}")
            # metric_dict thingy

            logger.warning(f"All reduced metrics: {...}")
            if consecutive_nan_count > 2 and not config.multidistillation.enabled:
                msg = "Too many consecutive nans detected in loss, avorting ..."
                logger.error(msg)
                raise RuntimeError(msg)
        else:
            consecutive_nan_count = 0

        ema_params_fsdp = update_ema(ema_params_fsdp, params_fsdp, mom)

        if (
            config.gram.use_loss
            and model.gram_rep_update
            and (it + 1) >= model.gram_init_first_update
            and (it + 1) % model.gram_update_frequency == 0
            and (config.gram.max_updates is None or num_gram_updates < config.gram.max_updates)
        ):
            logger.info(f"Updating Gram teacher from EMA teacher after iteration {it}")
            model.update_gradm()
            num_gram_updates += 1
        

        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(total_loss=total_loss, **metrics_dict)

        if (
            config.evaluation.eval_period_iterations > 0 and (iteration + 1) % config.evaluation.eval_period_iterations == 0
        ):
            do_test(config, model, f"training_{iteration}")

        
        if (iteration + 1) % config.checkpointing.period == 0:
            save_checkpoint(
                ckpt_dir / str(iteration),
                iteration=iteration,
                model=params_fsdp,
                optimizer=optimizer_state,
                overwrite=True,
            )
            if jax.process_index() == 0:
                keep_last_n_checkpoints(ckpt_dir, config.checkpointing.max_to_keep)
                if "keep_every" in config.checkpointing and (iteration + 1) % config.checkpointing.keep_every == 0:
                    keep_checkpoint_copy(ckpt_dir / str(iteration))

        iteration += 1


    jnp.array([1]).block_until_ready()
    jax.profiler.stop_trace()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




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