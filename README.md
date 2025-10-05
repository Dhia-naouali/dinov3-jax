# dinov3-jax [in progress]


This repository contains a Flax/JAX implementation of DINOv3 ([paper](https://arxiv.org/abs/2508.10104), [original repo](https://github.com/facebookresearch/dinov3)), originally developed in PyTorch by [Meta AI](https://github.com/facebookresearch)


<h2> Overview</h2>
this a re-implementation of Dinov3 by Meta, suing their original <a href="https://github.com/facebookresearch/dinov3">repo</a> in PyTorch mainly to have a better benchmark (using JAX: trading principles for optimization) and for learning puposes: SSL, distributed training bleeding edge training tricks and techniques ...


<h2> where it differs from the original repo</h2>
due to the differences in how PyTorch and JAX/flax are designed few design differences occured in this repo

<h4> distributed computation & communication</h4>
in a distributed setup Pytorch assigns a single process per device with a global process index for communication & orchestration, JAX on the other hand assigns a single process per host which will manage multiple devices using local communication, butter observed in multi-host multi-devices setups but in our case it's most significant in how data reaches devices instead of each device using a different set of workers to fetch it's chunk of the batch, using JAX each host will fetch / collect the whole data then shard / distribute it to it's devices


<h4> Activations checkpointing</h4>
the reference implementation used explicit checkpointing in two fashions: global & selective
in our implementation we decided to not enforce activation checkpoining and instead rely on the underlying compiler (XLA) since it has a global view of the computation graph and a set of heuristic on what to store / save and what to recompute during grads computation in the backward passes (adding a stricter checkpointing option would be just wrapping target modules in jax.checkpoint/jax.remat) 

<h4> FSDP</h4>
compared to the quite mature and (mostly) stable PyTorch implementation, JAXon the other hand doesn't have references, docs or materials on explicit FSDP implementations other than some heuristics and recommendations on how to shard params (except for a single docs page whispering FSDP in lower case and a legendary uni professor from amesterdam providing a reference implementation for an older version of JAX)
to achieve a PyTorch-like FSDP implementation we built an FSDP wrapper to be used around flax modules which will intercept computation to collect params and later on reshard both params and activations after the internal op(s)


<h4> Data loading</h4>
we used PyTorch's data loaders without pinned memory: JAX asynch dispatcher will take care of the equivalent, no multiple workers (`num_workers`) since it's a single process run by the host that will later on shard / distribute the batch on it's devices in a data parallel fashion


<h4> Checkpointing (model & optimizer)</h4>
the reference PyTorch implementation (multi-host) uses `dcp` (torch.distributed.checkpoint) API + `tempfiles`, in the JAX ecosystem orbax provide similar functionalities along with extra pre-implemented utils, to keep both implementations similar we only used high level / simple orbax APIs, for partial checkpointing (`register_dont_save_hooks`) not to save the forzen backbone each step, we simply pass the head(s) pytree to `save_checkpoint` given JAX/flax params are already being tossed around here and there and are always at reach


<h4> other minor tweaks</h4>
few other changes were introduced to avoid conflicts, function names where kept as similar as possible (if kept in the first place)



<p align="center">
  <img src="assets/dinov3-jax-run.png" width="960"><br>
  <em>few distributed training steps run profile (on 8 cpu cores:''') )</em>
</p>

<p>
  looking for a way to benchmark it against the reference PyTorch implementation (kaggle envs aren't compatible with JAX 0.7.1 (python 3.10)
</p>
<br>
<br>
<p align="center">
  <img src="assets/dino_at_home.png" width="420"><br>
  <em>"we have dino at home !"</em>
</p>


```text
@misc{siméoni2025dinov3,
      title={DINOv3}, 
      author={Oriane Siméoni and Huy V. Vo and Maximilian Seitzer and Federico Baldassarre and Maxime Oquab and Cijo Jose and Vasil Khalidov and Marc Szafraniec and Seungeun Yi and Michaël Ramamonjisoa and   Francisco Massa and Daniel Haziza and Luca Wehrstedt and Jianyuan Wang and Timothée Darcet and Théo Moutakanni and Leonel Sentana and Claire Roberts and Andrea Vedaldi and Jamie Tolan and John Brandt and Camille Couprie and Julien Mairal and Hervé Jégou and Patrick Labatut and Piotr Bojanowski},
      year={2025},
      eprint={2508.10104},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.10104}, 
}
```


