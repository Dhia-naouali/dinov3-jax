import logging
from collections import defaultdict

from flax.traverse_util import flatten_dict

logger = logging.getLogger("dinov3")



def get_params_groups_with_decay_fsdp(
        model, 
        lr_decay_rate=1., 
        patch_embed_lr_mult=1.,
        dino_head_wd_multiplier=1.
):
    chunked_blocks = False

    n_blocks = len([k for k in model.keys() if k.startswith("block")])
    is_backbone = "blocks_0" in model.keys()

    all_param_groups = []
    for name, param in flatten_dict(model, sep=".").items():
        # remove fsdp thingy from name
        # no need to fileter, only params make it here
        decay_rate = get_vit_lr_decay_rate(
            param,
            lr_decay_rate,
            num_layers=n_blocks,
            force_is_backbone=n_blocks > 0,
            chunked_blocks=chunked_blocks
        )
        d = {
            "name": name,
            "params": param,
            "is_last_layer": False,
            "lr_multiplier": decay_rate,
            "wd_multiplier": 1.
        }

        if "dino_head" in name:
            d["wd_multiplier"] = dino_head_wd_multiplier
        
        if "last_layer" in name:
            d["is_last_layer"] = True
        
        if name.endswith("bias") or "norm" in name or "gamma" in name or "fourier_w" in name:
            d["wd_multiplier"] = 0.
        
        if name == "patch_embed" in name:
            d["lr_multiplier"] *= patch_embed_lr_mult
        

        all_param_groups.append(d)
        logger.info(f"{name}: lr_multiplier: {d['lr_multiplier']}, wd_multiplier: {d['wd_multiplier']}")

    return all_param_groups




def get_vit_lr_decay_rate(
    name,
    lr_decay_rate=1.0,
    num_layers=12,
    force_is_backbone=False,
    chunked_blocks=False,
):
    print("temp lr decay rate: 1")
    return 1 # temp
    layer_id = num_layers + 1
    if name.startswith("backbone") or force_is_backbone:
        if (
            ".pos_embed" in name
            or ".patch_embed" in name
            or ".mask_token" in name
            or ".cls_token" in name
            or ".storage_tokens" in name
        ):
            layer_id = 0
        elif force_is_backbone and (
            "pos_embed" in name
            or "patch_embed" in name
            or "mask_token" in name
            or "cls_token" in name
            or "storage_tokens" in name
        ):
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
        elif chunked_blocks and "blocks." in name and "residual." not in name:
            layer_id = int(name[name.find("blocks.") :].split(".")[2]) + 1
        elif "blocks." in name and "residual." not in name:
            layer_id = int(name[name.find("blocks.") :].split(".")[1]) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)




def fuse_params_groups(all_params_groups, keys=("lr_multiplier", "wd_multiplier", "is_last_layer")):
    # fused / grouping params with similar configs (wd, lr, ...) together
    fused_params_groups = defaultdict(lambda: {"params":[]})
    for d in all_params_groups:
        id_ = ""
        for k in keys:
            id_ += k + str(d[k])

        # if not id_ in fused_params_groups: ?
        for k in keys:
            fused_params_groups[id_][k] = d[k]
        fused_params_groups[id_]["params"].append(d["params"])

    return fused_params_groups.values()



# Optimizer Parameter Groups Skeleton (JAX/Optax)
def get_param_groups(params):
    # TODO: Implement parameter grouping for optimizer
    return [{'params': params}]

