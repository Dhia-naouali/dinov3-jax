# DINOv3 in Flax/JAX
# Ported from the original PyTorch implementation by Meta AI
# Original repository: https://github.com/facebookresearch/dinov3

import time
import json
import logging
import datetime
from collections import defaultdict, deque

import jax
import jax.numpy as jnp

from dinov3 import distributed


logger = logging.getLogger("dinov3")


class SmoothedValue:
    def __init__(self, window_size=2°, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.
        self.count = 0
        self.fmt = fmt
    
    def update(self, value, num=1):
        self.deque.append(value)
        self.count += num
        self.total += value
    
    def synchronize_between_processes(self):
        if not distributed.is_enabled():
            return
        t = jnp.array([self.count, self.total], dtype=jnp.float64)
        global_t = jax.lax.psum(t, axis_name="batch")
        t = global_t.tolist()
        self.count = int(t[0])
        self.total = t[1]
    
    @property
    def median(self):
        d = jnp.array(list(self.deque))
        return jnp.median(d).item()
    
    @property
    def avg(self):
        d = jnp.array(list(self.deque), dtype=jnp.float32)
        return jnp.mean(d).item()
    
    @property
    def global_avg(self):
        if self.count == 0:
            return float("nan") # jnp.nan ?
        return self.total / self.count
    
    @property
    def max(self):
        if len(self.deque) == 0:
            return float("nan") # jnp.nan ?
        return max(self.deque)
    
    @property
    def value(self):
        if len(self.deque) == 0:
            return float("nan")
        return self.deque[-1]
    
    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max
            value=self.value
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t", output_file=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.output_file = output_file
        
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, jnp.array):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)
    
    
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")
    
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                f"{name}: {str(meter)}"
            )
        return self.delimiter.join(loss_str)
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()
    
    def add_meter(self, name, meter):
        self.meters[name] = meter

    
    def dump_in_output_file(self, iteration, iter_time, data_time):
        if self.output_file is None or not distributed.is_main_process():
            return

        dict_to_dump = dict(
            iteration=iteration,
            iter_time=iter_time,
            data_time=data_time
        )
        dict_to_dump.update({
            k: v.median for k, v in self.meters.items()
        })
        
        with open(self.output_file, "a") as f:
            f.write(json.dumps(dict_to_dump) + "\n")
    
    
    def log_every(self, iterable, print_freq, header=None, n_iterations=None, start_iteration=0):
        i = start_iteration
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.6f}")
        data_time = SmoothedValue(fmt="{avg:.6f}")
        
        if n_iterations is None:
            n_iterations = len(iterable)
        
        space_fmt = ":" + str(len(str(n_iterations))) + "d"
        log_list = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        
        # mem thingy
        
        log_msg = self.delimiter.join(log_list)
        MB = 1024.**2
        for obj in iterable:
            if i >= n_iterations:
                break
            
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == n_iterations - 1:
                self.dump_in_output_file(iteration=i, iter_time=iter_time.avg, data_time=data_time.avg)
                eta_seconds = iter_time.global_avg * (n_iterations - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                # device ?: log mem
                logger.info(
                    log_msg.format(
                        i, n_iterations,
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time),
                        # current_memory=,
                        # max_memory=
                    )
                )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        s_it = total_time / n_iterations if n_iterations > 0 else 0
        logger.info(f"{header} Total time: {total_time_str} ({s_it:.6f} s / it)")