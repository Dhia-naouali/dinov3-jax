import subprocess
import jax.numpy as jnp
from checkpointer import save_checkpoint, load_checkpoint

def test():
    model_params = {
        "layer1": {"w": jnp.ones((2, 2)), "b": jnp.zeros(2)},
        "layer2": {"w": jnp.ones((2, 3))*2, "b": jnp.zeros(3)}
    }
    optimizer_state = {
        "layer1": {"w": jnp.zeros((2, 2)), "b": jnp.zeros(2)},
        "layer2": {"w": jnp.zeros((2, 3)), "b": jnp.zeros(3)}
    }

    tmp_dir = "checkpoint_0"
    save_checkpoint(tmp_dir, params=model_params, optimizer_state=optimizer_state, iteration=5, overwrite=True)
    
    # load optimizer_state, strict_loading=True
    checkpoint = load_checkpoint(
        tmp_dir,
        abstract_model_params=model_params,
        abstract_optimizer_state=optimizer_state,
        strict_loading=True
    )
    print("optimizer, strict:", checkpoint)

    # load without optimizer_state, strict_loading=False
    checkpoint = load_checkpoint(
        tmp_dir,
        abstract_model_params=model_params,
        abstract_optimizer_state=None,
        strict_loading=False
    )
    print("without optimizer, partial:", checkpoint)

    checkpoint = load_checkpoint(
        tmp_dir,
        abstract_model_params={"layer1": model_params["layer1"]},
        abstract_optimizer_state=optimizer_state,
        strict_loading=False
    )
    print("load partial params:", checkpoint)


if __name__ == "__main__":
    test()
    subprocess.run("rm -r checkpoint_*", shell=True)