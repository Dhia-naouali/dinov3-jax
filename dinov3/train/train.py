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

# from somewhere import distributed


# logdir = "/tmp/jax_trace"
# os.makedirs(logdir, exist_ok=True)

# # Start recording a profile
# jax.profiler.start_trace(logdir)



logger = logging.getLogger("dinov3")
jax.config.update('jax_num_cpu_devices', 8)
INIT_PHASE = False

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
        print(k)
        if "student_" not in k:
            param_groups[f"student_{k}"] = param_groups.pop(k)
        print(param_groups.keys())

    lr_schedule = jnp.array(lr_schedule.gen())
    wd_schedule = jnp.array(wd_schedule.gen())
    last_layer_lr_schedule = jnp.array(last_layer_lr_schedule.gen())
    print()


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

    return optimizers



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
    # return None
    return optax.adamw(schedule, b1=config.optim.adamw_beta1, b2=config.optim.adamw_beta2)
    return None
    # for params_group in params_groups:
        # extract schedule
        # init optimizer
    
    # return optimizers




    # return None
    # return optax.adamw(schedule, b1=config.optim.adamw_beta1, b2=config.optim.adamw_beta2)




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
    print(args.eval_only)
    if args.eval_only:
        iteration = model.get_checkpointer_class()(
            model, save_dir=config.train.output_dir
        ).resume_or_load(
            config.MODEL.WEIGHTS, resume = not args.no_resume
        ).get("teration", 1) + 1
        
        return do_test(config, model, f"manual_{iteration}")
    do_train(config, model, resume=not args.no_resume)


def do_train(config, model, resume=False):
    class TrainState(train_state.TrainState):
        rngs: Any = None


    data_loader = build_multi_resolution_data_loader_from_cfg(
        config=config,
        model=model,
        start_iter=0,
    )
    fake_batch = next(iter(data_loader))

    batch_pspec = {
        'collated_global_crops': P("dp"),
        'collated_local_crops': P("dp"),
        "collated_masks": P("dp"),
        "mask_indices_list": P(),
        "n_masked_patches": P(),
        "masks_weight": P(),
        "upperbound": P(),
        # "global_batch_size": P(),
    }


    main_rng = jax.random.PRNGKey(config.train.seed)
    main_rng, init_rng, dropout_rng, drop_path_rng = jax.random.split(main_rng, 4)
    # no process subgroups
    ckpt_dir = Path(config.train.output_dir, "ckpt").expanduser()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def init_dp(rng, inputs, model):
        init_rng, rng = jax.random.split(rng)
        return model.init(init_rng, inputs, teacher_temp=.7, iteration=0, init_phase=True) # somehow state


    mesh = jax.make_mesh(
        (len(jax.devices()),),
        ("dp",)
    )
    def shard_batch_item(item, spec):
        return jax.device_put(item, jax.sharding.NamedSharding(mesh, spec))

    fake_batch = jax.tree_util.tree_map(shard_batch_item, fake_batch, batch_pspec)


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
            fake_batch,
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

    params_fsdp = init_fsdp(init_rng, fake_batch)

    # init_params = model.init(key, fake_batch, teacher_temp=.7, iteration=0, init_phase=True)

    param_groups = model.get_params_groups(params_fsdp["params"])
    student_params = {
        k: v for k, v in params_fsdp["params"].items() if "student_" in k
    }
    print("student_params", student_params.keys())
    print("param_group", param_groups.keys())
    # assert set(student_params.keys()) == set(param_groups.keys()), "param_groups must match student_params keys"

    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule
    ) = build_schedulers(config)
    # import IPython; IPython.embed()
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


    logger.info(f"Starting training from iteration {start_iter}")
    metrics_file = os.path.join(config.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    # gc.disable()
    gc.collect()

    # next(iter(data_loader))
    rngs={"dropout": jax.random.PRNGKey(1), "drop_path": jax.random.PRNGKey(2)}


    def train_step(
            params,
            batch,
            optimizer_state,
            teacher_temp,
            iteration,
            root_rngs,
            axis_name="dp"
    ):
        student_params = {k: v for k, v in params["params"].items() if "student_" in k}
        print(f"entry: {params['params'].keys()}")
        print(f"entry student: {student_params.keys()}")
        axis_idx = jax.lax.axis_index(axis_name)
        rngs = {k: jax.random.fold_in(v, axis_idx) for k, v in root_rngs.items()}
        def loss_fn(student_params):
            temp_params = dict(params)
            temp_params["params"] = {
                k: student_params[k] if "student_" in k else v
                for k, v in temp_params["params"].items()
            }
            loss = model.apply(temp_params, batch, teacher_temp=teacher_temp, iteration=iteration, rngs=rngs)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(student_params)
        print(f"grads: {grads.keys()}")
        
        # grads norm

        # grads = sync_grads(grads)
        # plain arrays for optimizer
        student_params_plain = jax.tree_util.tree_map(
            lambda x: x.value if isinstance(x, nn.Partitioned) else x, student_params
        )

        # partitioned params for forward/backward
        student_params_partitioned = student_params
        updates, optimizer_state = optimizer.update(grads, optimizer_state, student_params_plain)
        student_params_plain = optax.apply_updates(student_params_plain, updates)
        student_params_partitioned = jax.tree_map(
            lambda x, y: nn.Partitioned(x, mesh=x.mesh if isinstance(x, nn.Partitioned) else mesh, names=x.names if isinstance(x, nn.Partitioned) else ("dp",)),
            student_params_plain,
            student_params_partitioned
        )
        raise Exception()
        print(f"params: {params.keys()}")
        print(f"student_params: {student_params.keys()}")
        print(f"params_params: {params['params'].keys()}")
        return params, optimizer_state, loss


    optimizer_specs = nn.get_partition_spec(optimizer_state)


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
            out_specs=(param_specs, optimizer_specs, P())
        ),
        donate_argnums=(0, 1, 2)
    )

    train_step_fsdp(params_fsdp, fake_batch, optimizer_state, teacher_temp_schedule[12], 1, rngs)
    import IPython; IPython.embed()

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


    def train_step(state, batch):
        # rng, _ = jax.random.split(state.rngs)
        loss, grads = jax.value_and_grad(state.apply_fn)(
            state.params, 
            batch, 
            teacher_temp=.7, 
            iteration=0,
            rngs=state.rngs,
        )
        grads = sync_grads(grads)

        new_state = state.apply_gradients(grads=grads)
        loss = jax.lax.pmean(loss, axis_name="dp")
        return new_state, loss


    train_step_fsdp = jax.jit(
        jax.shard_map(
            train_step,
            mesh=mesh,
            in_specs=...,
            out_specs=...
        )
    )



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

        import IPython; IPython.embed()
        train_loss, metrics_dict = model.apply(params_fsdp, data, teacher_temp=teacher_temp, iteration=it, rngs={
            "dropout": jax.random.PRNGKey(1), "drop_path": jax.random.PRNGKey(2)
        })


        if config.optim.clip_grad:
            print("to clip grads")


        import IPython; IPython.embed()

        # reduce loss & metric logs
        total_loss_all_ranks = ...


        if total_loss_all_ranks.isnan().any():
            ...
        else:
            consecutive_nan_count = 0


        optimizer.apply(...)
        model.update_ema(mom)


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
