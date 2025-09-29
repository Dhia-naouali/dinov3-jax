import logging
from collections import defaultdict
from typing import Union

import jax
from flax.traverse_util import flatten_dict, unflatten_dict

logger = logging.getLogger("dinov3")


class ParamDict:
    def __init__(
            self, 
            name: str = None,
            is_last_layer: bool = None,
            lr_multiplier: Union[int, float] = None,
            wd_multiplier: Union[int, float] = None,
            foreach: bool = None,
            fused: bool = None
        ):
        self.name = name
        self.is_last_layer = is_last_layer
        self.lr_multiplier = lr_multiplier
        self.wd_multiplier = wd_multiplier
        self.foreach = foreach
        self.fused = fused

    def __repr__(self):
        return "FIXED[" + repr(dict(vars(self))) + "]"
    
    def __str__(self):
        return "FIXED[" + str(dict(vars(self))) + "]"

    def __eq__(self, other):
        return vars(self) == vars(other)

    def __hash__(self):
        return hash(tuple(vars(self).values()))

# to be used later to construct the params mask for the multi-optimizers thingy
# jax.tree_util.register_pytree_node(
#     ParamDict,
#     lambda x: (tuple(x.dict.values()), tuple(x.dict.keys())),
#     lambda values, keys : ParamDict(dict(zip(keys, values))),
# )


def get_params_groups_with_decay_fsdp(
        model, 
        lr_decay_rate=1., 
        patch_embed_lr_mult=1.,
        dino_head_wd_multiplier=1.,
        root_name=""
):
    chunked_blocks = False
    n_blocks = len([k for k in model.keys() if k.startswith("block")])

    all_param_groups = {}
    for name, param in flatten_dict(model, sep=".").items():
        decay_rate = get_vit_lr_decay_rate(
            name,
            lr_decay_rate,
            num_layers=n_blocks,
            force_is_backbone=n_blocks>0,
            chunked_blocks=chunked_blocks,
            root_name=root_name
        )

        d = {
            # "name": name,
            "is_last_layer": False,
            "lr_multiplier": decay_rate,
            "wd_multiplier": 1.,
        }

        if "dino_head" in name:
            d["wd_multiplier"] = dino_head_wd_multiplier

        if "last_layer" in name:
            d["is_last_layer"] = True

        if name.endswith("bias") or "norm" in name or "gamma" in name or "fourier_w" in name:
            d["wd_multiplier"] = 0

        if "patch_embed" in name:
            d["lr_multiplier"] *= patch_embed_lr_mult
        
        all_param_groups[name] = ParamDict(**d)
        logger.info(f"{name}: lr_multiplier: {d['lr_multiplier']}, wd_multiplier: {d['wd_multiplier']}")

    return unflatten_dict(all_param_groups, sep=".")




def get_vit_lr_decay_rate(
    name,
    lr_decay_rate=1.0,
    num_layers=12,
    force_is_backbone=False,
    chunked_blocks=False,
    root_name=""
):
    name = ".".join([root_name, name])
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
        elif "blocks" in name and "residual" not in name:
            layer_id = int(name.split("blocks_")[1].split(".")[0]) + 1
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def fuse_params_groups(all_params_groups, keys=("lr_multiplier", "wd_multiplier", "is_last_layer"), root_name=""):

    def auto_counter():
      counter = {"group": 0}
      def _next():
          counter["group"] += 1
          return f"{root_name}_group_{counter['group']}"
      return _next
    next_id = auto_counter()
    dd = defaultdict(next_id)

    def fn(d):
        # d = vars(d)
        # id_ = ""
        # for k in d:
        #     id_ += k + str(d[k]) + "_"
        return dd[d]
    

    fused_params_groups = jax.tree_util.tree_map(fn, all_params_groups)
    fused_params_groups["--groups--"] = {
        v: k for k, v in dd.items()
    }
    return fused_params_groups


    # fused / grouping params with similar configs (wd, lr, ...) together
    fused_params_groups = defaultdict(lambda: {"params":[]})
    for d in all_params_groups:
        id_ = ""
        for k in keys:
            id_ += k + str(d[k])

        # if not id_ in fused_params_groups: ?
        for k in keys:
            fused_params_groups[id_][k] = d[k]
        fused_params_groups[id_]["group_name"] = d["name"]
        fused_params_groups[id_]["params"].append(d["params"])

    return fused_params_groups.values()