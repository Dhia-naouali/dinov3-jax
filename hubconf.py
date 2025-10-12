import re
import os
import torch

import jax
import flax
import flax.linen as nn
import jax.numpy as jnp

from dinov3_jax.models.vision_transformer import vit_small


# init jax model and params/checkpoint
jax_model = vit_small(layerscale_init=1., n_storage_tokens=4, use_fsdp=False)
dummy_input = jnp.zeros((2, 224, 224, 3))
jax_checkpoint = jax_model.init(jax.random.key(12), dummy_input)



# init torch model and params/checkpoint
model_name = "dinov3_vits16"
torch_model = torch.hub.load(
    repo_or_dir="facebookresearch/dinov3",
    model=model_name,
    source="github",
    pretrained=False
)

torch_checkpoint = torch_model.state_dict()


# implementation detials: in the original implementation a "bias_mask" were used for qkv along with "bias" (which is all zeros)
# which wasn't used in our implementation
# torch_checkpoint["blocks.0.attn.qkv.bias"].any(), torch_checkpoint["blocks.0.attn.qkv.bias"].sum()

flat = flax.traverse_util.flatten_dict(jax_checkpoint["params"], sep=".")
t2j = lambda x: jnp.array(x.detach().cpu().numpy())


def setup_checkpoint(torch_params, jax_params):
    def mapper(tk):
        t = False
        jk = tk
        if "weight" in tk:
            if "norm" in tk.split(".")[-2]:
                weight_eq = "scale"
            else:
                weight_eq = "kernel"
                t = True
            jk = jk.replace("weight", weight_eq)

        jk = re.sub(
            r"fc(\d+)",
            lambda n: f"Dense_{int(n.group(1)) - 1}",
            jk
        )
        
        jk = jk.replace("blocks.", "blocks_")        
        return jk, t

    
    out = {}
    torch_blocks = [k for k in torch_params.keys()]
    jax_blocks = [k for k in jax_params.keys()]
    print(len(torch_blocks), len(jax_blocks))
    for tb_item in torch_blocks:
        if "bias_mask" in tb_item:
            continue
        jb_item, t = mapper(tb_item)
        out[jb_item] = t2j(torch_params[tb_item].T) if t else t2j(torch_params[tb_item])
    return out


checkpoint = setup_checkpoint(torch_checkpoint, flat)
jax_checkpoint["params"] = flax.traverse_util.unflatten_dict(checkpoint, sep=".")
jax_checkpoint["constants"]["rope_embed"] = jax_checkpoint["params"].pop("rope_embed")


# jax_checkpoint is ready to go (shape wise)
y = jax_model.apply(jax_checkpoint, dummy_input)