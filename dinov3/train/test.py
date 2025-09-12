from dinov3.models import build_model_from_cfg


build_model_from_cfg(cfg)



def setup_job(
        output_dir=None,
        distributed_enabled=True,
        logging_enabled=True,
        seed=12,
        restricted_print_to_main_process=True,
        distributed_timeout=None
):
    if output_dir is not None:
        output_dir = os.path.relapath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    if logging_enabled:
        setup_logging(
            output=output_dir,
            level=logging.INOF,
            log_to_stdout_only_in_main_process=restricted_print_to_main_process,
        )
    
    assert not distributed_enabled # temp
    # if distributed_enabled:
    #     distributed.enable(
    #         overwrite=True,

    #     )

    if seed is not None:
        rank = distributed.get_rank()
        fix_random_seeds(seed + rank)
    
    logger = logging.getLogger("dinov3")
    logger.info(f"git:\n    {get_sha()}\n")

    # conda thingies ...
    