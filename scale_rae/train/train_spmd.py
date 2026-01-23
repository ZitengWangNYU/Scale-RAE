import os
import random
import functools
import numpy as np
from datetime import timedelta, datetime
import torch.distributed as dist


_LOCAL_PROCESS_GROUP = None
import gcsfs
fs = gcsfs.GCSFileSystem(project=os.getenv('GCP_PROJECT', 'your-gcp-project'))


# Setup logger with caching for once-only messages
import logging
@functools.lru_cache(None)
def warning_once(self, *args, **kwargs):
    """
    This method is identical to `logger.warning()`, but will emit the warning with the same message only once

    Note: The cache is for the function arguments, so 2 different callers using the same arguments will hit the cache.
    The assumption here is that all warning messages are unique across the code. If they aren't then need to switch to
    another type of cache that includes the caller frame information in the hashing function.
    """
    self.warning(*args, **kwargs)

@functools.lru_cache(None)
def info_once(self, *args, **kwargs):
    """
    This method is identical to `logger.info()`, but will emit the info with the same message only once

    Note: The cache is for the function arguments, so 2 different callers using the same arguments will hit the cache.
    The assumption here is that all warning messages are unique across the code. If they aren't then need to switch to
    another type of cache that includes the caller frame information in the hashing function.
    """
    self.info(*args, **kwargs)

logging.Logger.warning_once = warning_once
logging.Logger.info_once = info_once

logger = logging.getLogger(__name__)

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
import torch_xla.distributed.parallel_loader as pl
import torch.distributed as dist
from collections import defaultdict

# Import to register the `xla://` init_method
import torch_xla.distributed.xla_backend

from io import BytesIO
import zstandard as zstd

################################################################################
# Monkey-patch accelerate's source code
from torch.utils.data import BatchSampler, DataLoader, IterableDataset

class MpDeviceLoaderWrapper(pl.MpDeviceLoader):
    def __init__(self, dataloader, device):
        logger.info("Calling monkey patch for MpDeviceLoaderWrapper...")
        input_sharding = {
            "input_ids": xs.ShardingSpec(xs.get_global_mesh(), ("fsdp", None), minibatch=True),
            "labels": xs.ShardingSpec(xs.get_global_mesh(), ("fsdp", None), minibatch=True),
            "attention_mask": xs.ShardingSpec(xs.get_global_mesh(), ("fsdp", None), minibatch=True),
            "vision_token_indices": xs.ShardingSpec(xs.get_global_mesh(), ("fsdp", None), minibatch=True),
            "images": xs.ShardingSpec(xs.get_global_mesh(), ("fsdp", None, None, None), minibatch=True),
            "answer_token_mask": xs.ShardingSpec(xs.get_global_mesh(), ("fsdp", None), minibatch=True),
            "answer_img_mask": xs.ShardingSpec(xs.get_global_mesh(), ("fsdp", None), minibatch=True),
            "reverse_vti": xs.ShardingSpec(xs.get_global_mesh(), ("fsdp", None), minibatch=True),
            # "images_2": xs.ShardingSpec(xs.get_global_mesh(), ("fsdp", None, None, None), minibatch=True),
            "images_gen": xs.ShardingSpec(xs.get_global_mesh(), ("fsdp", None, None, None), minibatch=True),
        }

        super().__init__(
            dataloader,
            device=xm.xla_device(),
            input_sharding=input_sharding,
            # loader_prefetch_size=min(dataloader.total_batch_size * 2, 64),
            # device_prefetch_size=min(dataloader.total_batch_size * 2, 64),
            # loader_prefetch_size=32,
            # device_prefetch_size=32,
            loader_prefetch_size=64,
            device_prefetch_size=32,
        )
        self._rng_types = self._loader.rng_types
        self._loader.rng_types = None

    def __iter__(self):
        from accelerate.utils import synchronize_rng_states
        if self._rng_types is not None:
            synchronize_rng_states(self._rng_types, self._loader.synchronized_generator)

        return super().__iter__()

    @property
    def total_batch_size(self):
        logger.info(f"total_batch_size: {self._loader.total_batch_size}")
        return self._loader.total_batch_size

    @property
    def total_dataset_length(self): 
        logger.info(f"total_dataset_length: {self._loader.total_dataset_length}")
        return self._loader.total_dataset_length



def skip_first_batches(dataloader, num_batches=0):
    """
    Creates a `torch.utils.data.DataLoader` that will efficiently skip the first `num_batches`.
    """
    logger.info("Calling monkey patch for skip_first_batches...")
    #################################### ! CHANGES HERE ! ####################################
    import accelerate
    from scale_rae.train.webdataset_trainer import WebDatasetLazySupervisedDataset

    # For our iterable tar-mode dataset, do NOT skip by consuming batches; resume is handled internally
    try:
        dataset = getattr(dataloader, "dataset", None)
        if isinstance(dataset, WebDatasetLazySupervisedDataset):
            logger.info("Iterable WebDataset detected: bypassing skip_first_batches (resume handled by dataset)")
            return dataloader
    except Exception:
        pass

    if isinstance(dataloader, MpDeviceLoaderWrapper):
        return MpDeviceLoaderWrapper(skip_first_batches(dataloader._loader, num_batches), None)
    #################################### ! CHANGES HERE ! ####################################

    dataset = dataloader.dataset
    sampler_is_batch_sampler = False
    if isinstance(dataset, IterableDataset):
        new_batch_sampler = None
    else:
        sampler_is_batch_sampler = isinstance(dataloader.sampler, BatchSampler)
        batch_sampler = dataloader.sampler if sampler_is_batch_sampler else dataloader.batch_sampler
        new_batch_sampler = accelerate.data_loader.SkipBatchSampler(batch_sampler, skip_batches=num_batches)

    # We ignore all of those since they are all dealt with by our new_batch_sampler
    ignore_kwargs = [
        "batch_size",
        "shuffle",
        "sampler", 
        "batch_sampler",
        "drop_last",
    ]

    kwargs = {
        k: getattr(dataloader, k, accelerate.data_loader._PYTORCH_DATALOADER_KWARGS[k])
        for k in accelerate.data_loader._PYTORCH_DATALOADER_KWARGS
        if k not in ignore_kwargs
    }

    # Need to provide batch_size as batch_sampler is None for Iterable dataset
    if new_batch_sampler is None:
        kwargs["drop_last"] = dataloader.drop_last
        kwargs["batch_size"] = dataloader.batch_size

    if isinstance(dataloader, accelerate.data_loader.DataLoaderDispatcher):
        if new_batch_sampler is None:
            # Need to manually skip batches in the dataloader
            kwargs["skip_batches"] = num_batches
        dataloader = accelerate.data_loader.DataLoaderDispatcher(
            dataset,
            split_batches=dataloader.split_batches,
            batch_sampler=new_batch_sampler,
            _drop_last=dataloader._drop_last,
            **kwargs,
        )
    elif isinstance(dataloader, accelerate.data_loader.DataLoaderShard):
        if new_batch_sampler is None:
            # Need to manually skip batches in the dataloader
            kwargs["skip_batches"] = num_batches
        elif sampler_is_batch_sampler:
            kwargs["sampler"] = new_batch_sampler
            kwargs["batch_size"] = dataloader.batch_size
        else:
            kwargs["batch_sampler"] = new_batch_sampler
        dataloader = accelerate.data_loader.DataLoaderShard(
            dataset,
            device=dataloader.device,
            rng_types=dataloader.rng_types,
            synchronized_generator=dataloader.synchronized_generator,
            **kwargs,
        )
    else:
        if new_batch_sampler is None:
            # Need to manually skip batches in the dataloader
            dataloader = accelerate.data_loader.SkipDataLoader(dataset, skip_batches=num_batches, **kwargs)
        else:
            dataloader = DataLoader(dataset, batch_sampler=new_batch_sampler, **kwargs)

    return dataloader

def prepare_data_loader(
    dataloader=None,
    device=None,
    num_processes=None,
    process_index=None,
    split_batches=False,
    put_on_device=False,
    rng_types=None,
    dispatch_batches=None,
    even_batches=True,
    slice_fn_for_dispatch=None,
):
    logger.info("Calling monkey patch for prepare_data_loader...")
    import accelerate
    import torch

    if dispatch_batches is None:
        if not put_on_device:
            dispatch_batches = False
        else:
            dispatch_batches = isinstance(dataloader.dataset, IterableDataset)

    if dispatch_batches and not put_on_device:
        raise ValueError("Using `dispatch_batches=True` requires `put_on_device=True`.")
    # Grab defaults from AcceleratorState
    state = accelerate.state.AcceleratorState()
    if num_processes is None:
        num_processes = state.num_processes
    if process_index is None:
        process_index = state.process_index

    #################################### ! CHANGES HERE ! ####################################
    num_processes = torch.distributed.get_world_size()
    process_index = torch.distributed.get_rank()
    #################################### ! CHANGES HERE ! ####################################

    # Sanity check
    if split_batches and dataloader.batch_size > 1 and dataloader.batch_size % num_processes != 0:
        raise ValueError(
            f"To use a `DataLoader` in `split_batches` mode, the batch size ({dataloader.batch_size}) "
            f"needs to be a round multiple of the number of processes ({num_processes})."
        )

    new_dataset = dataloader.dataset
    # Datasets like WebDataset may perform their own node/worker sharding.
    # If so, skip wrapping with IterableDatasetShard to avoid double-sharding.
    dataset_already_sharded = getattr(new_dataset, "already_sharded", False)
    # Iterable dataset doesn't like batch_sampler, but data_loader creates a default one for it
    new_batch_sampler = dataloader.batch_sampler if not isinstance(new_dataset, IterableDataset) else None
    sampler_is_batch_sampler = False
    synchronized_generator = None
    # No change if no multiprocess

    if (num_processes != 1 or state.distributed_type == accelerate.state.DistributedType.MEGATRON_LM) and not dispatch_batches and not dataset_already_sharded:
        if isinstance(new_dataset, IterableDataset):
            if getattr(dataloader.dataset, "generator", None) is not None:
                synchronized_generator = dataloader.dataset.generator
            new_dataset = accelerate.data_loader.IterableDatasetShard(
                new_dataset,
                batch_size=dataloader.batch_size,
                drop_last=dataloader.drop_last,
                num_processes=num_processes,
                process_index=process_index,
                split_batches=split_batches,
            )
        else:
            # New batch sampler for the current process.
            sampler_is_batch_sampler = isinstance(dataloader.sampler, torch.utils.data.BatchSampler)
            if sampler_is_batch_sampler:
                sampler = dataloader.sampler.sampler
            else:
                sampler = dataloader.batch_sampler.sampler
            if hasattr(sampler, "generator"):
                if sampler.generator is None:
                    sampler.generator = torch.Generator()
                synchronized_generator = sampler.generator

            batch_sampler = dataloader.sampler if sampler_is_batch_sampler else dataloader.batch_sampler
            new_batch_sampler = accelerate.data_loader.BatchSamplerShard(
                batch_sampler,
                num_processes=num_processes,
                process_index=process_index,
                split_batches=split_batches,
                even_batches=even_batches,
            )
    else:
        logger.info("No change to the dataloader")


    # We ignore all of those since they are all dealt with by our new_batch_sampler
    ignore_kwargs = [
        "batch_size",
        "shuffle",
        "sampler",
        "batch_sampler",
        "drop_last",
    ]

    if rng_types is not None and synchronized_generator is None and "generator" in rng_types:
        rng_types.remove("generator")

    kwargs = {
        k: getattr(dataloader, k, accelerate.data_loader._PYTORCH_DATALOADER_KWARGS[k])
        for k in accelerate.data_loader._PYTORCH_DATALOADER_KWARGS
        if k not in ignore_kwargs
    }

    # Need to provide batch_size as batch_sampler is None for Iterable dataset
    if new_batch_sampler is None:
        kwargs["drop_last"] = dataloader.drop_last
        kwargs["batch_size"] = (
            dataloader.batch_size // num_processes if split_batches and not dispatch_batches else dataloader.batch_size
        )

    if dispatch_batches:
        kwargs.pop("generator")
        dataloader = accelerate.data_loader.DataLoaderDispatcher(
            new_dataset,
            split_batches=split_batches,
            batch_sampler=new_batch_sampler,
            _drop_last=dataloader.drop_last,
            slice_fn=slice_fn_for_dispatch,
            **kwargs,
        )
    elif sampler_is_batch_sampler:
        dataloader = accelerate.data_loader.DataLoaderShard(
            new_dataset,
            device=device if put_on_device and state.distributed_type != accelerate.state.DistributedType.TPU else None,
            sampler=new_batch_sampler,
            batch_size=dataloader.batch_size,
            rng_types=rng_types,
            synchronized_generator=synchronized_generator,
            **kwargs,
        )
    else:
        dataloader = accelerate.data_loader.DataLoaderShard(
            new_dataset,
            device=device if put_on_device and state.distributed_type != accelerate.state.DistributedType.TPU else None,
            batch_sampler=new_batch_sampler,
            rng_types=rng_types,
            synchronized_generator=synchronized_generator,
            **kwargs,
        )

    if state.distributed_type == accelerate.state.DistributedType.TPU:
        return MpDeviceLoaderWrapper(dataloader, device)
    return dataloader


    

def accelerator_prepare_data_loader(
    self, data_loader, device_placement=None, slice_fn_for_dispatch=None
):
    logger.info("Calling monkey patch for Accelerator.prepare_data_loader...")
    import accelerate
    # Ensure we can't double wrap a DataLoader due to `find_batch_size`
    if getattr(data_loader, "_is_accelerate_prepared", False):
        if data_loader not in self._dataloaders:
            self._dataloaders.append(data_loader)
        return data_loader
    if device_placement is None:
        device_placement = self.device_placement if self.distributed_type != accelerate.state.DistributedType.TPU else False

    #################################### ! CHANGES HERE ! ####################################
    prepared_data_loader = prepare_data_loader(
        data_loader,
        self.device,
        num_processes=self.num_processes,
        process_index=self.process_index,
        split_batches=self.split_batches,
        put_on_device=device_placement,
        rng_types=self.rng_types.copy(),
        dispatch_batches=self.dispatch_batches,
        even_batches=self.even_batches,
        slice_fn_for_dispatch=slice_fn_for_dispatch,
    )
    #################################### ! CHANGES HERE ! ####################################

    self._dataloaders.append(prepared_data_loader)
    return prepared_data_loader

import accelerate
import accelerate.data_loader
accelerate.skip_first_batches = skip_first_batches
accelerate.data_loader.skip_first_batches = skip_first_batches
accelerate.data_loader.MpDeviceLoaderWrapper = MpDeviceLoaderWrapper
accelerate.accelerator.Accelerator.prepare_data_loader = accelerator_prepare_data_loader
################################################################################

################################################################################
# monkey-patched to transformers' source code
import transformers
from functools import partial
from torch.optim.lr_scheduler import LambdaLR

def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
):
    logger.info_once("Calling monkey patch for _get_cosine_schedule_with_warmup_lr_lambda...")
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

    min_lr_ratio = float(os.getenv("SCALE_RAE_MIN_LR_RATIO", "0."))

    scale_ratio = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    if min_lr_ratio > 0.0:
        logger.info_once(f"Using min_lr_ratio {min_lr_ratio} to scale the learning rate...")
        scale_ratio = min_lr_ratio + (1.0 - min_lr_ratio) * scale_ratio
    return scale_ratio

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    logger.info("Calling monkey patch for get_cosine_schedule_with_warmup...")
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

transformers.optimization.TYPE_TO_SCHEDULER_FUNCTION = {transformers.trainer_utils.SchedulerType.COSINE: get_cosine_schedule_with_warmup}
################################################################################

from transformers import Trainer
import time
import torch
import functools
from torch import nn
from packaging import version

from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_pt_utils import (
    get_module_class_from_name,
)
from transformers.training_args import ParallelMode
from transformers.utils import (
    logging,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_neuroncore_available,
)
from accelerate.utils import DistributedDataParallelKwargs

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

def _wrap_model(self, model, training=True, dataloader=None):
    logger.info("Calling monkey patch for Trainer._wrap_model...")
    self.is_fsdp_xla_v2_enabled = True

    if self.args.use_ipex:
        dtype = torch.bfloat16 if self.use_cpu_amp else torch.float32
        model = self.ipex_optimize_model(model, training, dtype=dtype)

    if is_sagemaker_mp_enabled():
        # Wrapping the base model twice in a DistributedModel will raise an error.
        if isinstance(self.model_wrapped, smp.model.DistributedModel):
            return self.model_wrapped
        return smp.DistributedModel(model, backward_passes_per_step=self.args.gradient_accumulation_steps)

    # train/eval could be run multiple-times - if already wrapped, don't re-wrap it again
    if self.accelerator.unwrap_model(model) is not model:
        return model

    # Mixed precision training with apex (torch < 1.6)
    if self.use_apex and training:
        model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)

    # Multi-gpu training (should be after apex fp16 initialization) / 8bit models does not support DDP
    if self.args.n_gpu > 1 and not getattr(model, "is_loaded_in_8bit", False):
        model = nn.DataParallel(model)

    if self.args.jit_mode_eval:
        start_time = time.time()
        model = self.torch_jit_model_eval(model, dataloader, training)
        self.jit_compilation_time = round(time.time() - start_time, 4)

    # Note: in torch.distributed mode, there's no point in wrapping the model
    # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
    if not training:
        return model

    # Distributed training (should be after apex fp16 initialization)
    # Distributed training using PyTorch FSDP
    if self.is_fsdp_xla_enabled:
        try:
            from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
            from torch_xla.distributed.fsdp import checkpoint_module
            from torch_xla.distributed.fsdp.wrap import (
                size_based_auto_wrap_policy,
                transformer_auto_wrap_policy,
            )

            if self.is_fsdp_xla_v2_enabled:
                from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
                    SpmdFullyShardedDataParallel as FSDPv2,
                )
        except ImportError:
            raise ImportError("Missing XLA FSDP related module; please make sure to use torch-xla >= 2.0.")
        auto_wrap_policy = None
        auto_wrapper_callable = None
        default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
        fsdp_transformer_layer_cls_to_wrap = self.args.fsdp_config.get(
            "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
        )


        print("Wrap Models are:", fsdp_transformer_layer_cls_to_wrap, flush=True)
        
        if self.args.fsdp_config["min_num_params"] > 0:
            auto_wrap_policy = functools.partial(
                size_based_auto_wrap_policy, min_num_params=self.args.fsdp_config["min_num_params"]
            )
        elif fsdp_transformer_layer_cls_to_wrap is not None:
            transformer_cls_to_wrap = set()
            for layer_class in fsdp_transformer_layer_cls_to_wrap:
                transformer_cls = get_module_class_from_name(model, layer_class)
                if transformer_cls is None:
                    raise Exception("Could not find the transformer layer class to wrap in the model.")
                else:
                    transformer_cls_to_wrap.add(transformer_cls)

            auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                # Transformer layer class to wrap
                transformer_layer_cls=transformer_cls_to_wrap,
            )
        fsdp_kwargs = self.args.xla_fsdp_config
        if self.args.fsdp_config["xla_fsdp_grad_ckpt"]:
            if model.config.use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
                )
                model.config.use_cache = False

            # Apply gradient checkpointing to auto-wrapped sub-modules if specified
            def auto_wrapper_callable(m, *args, **kwargs):
                target_cls = FSDP if not self.is_fsdp_xla_v2_enabled else FSDPv2
                return target_cls(checkpoint_module(m), *args, **kwargs)

        #################################### ! CHANGES HERE ! ####################################
        # Wrap the base model with an outer FSDP wrapper
        if self.is_fsdp_xla_v2_enabled:

            def shard_output(output, mesh):
                from transformers.modeling_outputs import CausalLMOutputWithPast

                real_output = None
                if isinstance(output, torch.Tensor):
                    real_output = output
                elif isinstance(output, tuple):
                    real_output = output[0]
                elif isinstance(output, CausalLMOutputWithPast):
                    real_output = output.logits

                if real_output is None:
                    raise ValueError("Something went wrong, the output of the model shouldn't be `None`")

                if len(real_output.shape) == 3:
                    xs.mark_sharding(real_output, mesh, ("fsdp", None, None))
                elif len(real_output.shape) == 2:
                    xs.mark_sharding(real_output, mesh, ("fsdp", None))
                else:
                    raise ValueError(f"Unexpected output shape: {real_output.shape}")
                # xs.mark_sharding(real_output, mesh, ("fsdp", None, None))

            self.model = model = FSDPv2(
                model,
                shard_output=shard_output,
                auto_wrap_policy=auto_wrap_policy,
                auto_wrapper_callable=auto_wrapper_callable,
            )

            from torch_xla.distributed.fsdp.utils import apply_xla_patch_to_nn_linear
            # Use a patch to `nn.Linear` (`torch.nn.functional.linear`) in XLA so that its
            # backward pass will use its weight parameter rather than an intermediate result.
            # (see https://github.com/pytorch/xla/issues/3811 for details)
            self.model = apply_xla_patch_to_nn_linear(self.model, xs.xla_patched_nn_linear_forward)
            #################################### ! CHANGES HERE ! ####################################
            # from torch_xla.distributed.spmd.xla_sharding import apply_xla_patch_to_nn_linear
            # from scale_rae.train.shard_model import shard_torch_xla_model_from_config, wrap_module

            # def maybe_checkpoint(mod, _name):
            #     if isinstance(mod, tuple(transformer_cls_to_wrap)):
            #         logger.info(f"Applying gradient checkpointing to {_name}...")
            #         return checkpoint_module(mod)
            #     return mod

            # def maybe_add_barrier(mod, _name):
            #     if isinstance(mod, tuple(transformer_cls_to_wrap)):
            #         # Register a backward hook to place optimization barrier to prevent
            #         # gigantic fusions on syncing the gradients.
            #         logger.info(f"Adding optimization barrier to {_name}...")
            #         xs.apply_backward_optimization_barrier(mod)
            #         return mod
            #     return mod

            # model = model.to("xla")
            # model = apply_xla_patch_to_nn_linear(model)
            
            # for name, param in model.named_parameters():
            #     print(name, param.shape)

            # sharding_config = {
            #     # language model
            #     "model.embed_tokens.weight": ["fsdp", None],
            #     "model.layers.*.self_attn.q_proj.weight": ["fsdp", None],
            #     "model.layers.*.self_attn.q_proj.bias": ["fsdp",],
            #     "model.layers.*.self_attn.k_proj.weight": [None, "fsdp"],
            #     "model.layers.*.self_attn.k_proj.bias": ["fsdp",],
            #     "model.layers.*.self_attn.v_proj.weight": [None, "fsdp"],
            #     "model.layers.*.self_attn.v_proj.bias": ["fsdp",],
            #     "model.layers.*.self_attn.o_proj.weight": ["fsdp", None],
            #     "model.layers.*.mlp.gate_proj.weight": ["fsdp", None],
            #     "model.layers.*.mlp.up_proj.weight": ["fsdp", None],
            #     "model.layers.*.mlp.down_proj.weight": [None, "fsdp"],
            #     "model.layers.*.input_layernorm.weight": ["fsdp",],
            #     "model.layers.*.post_attention_layernorm.weight": ["fsdp",],
            #     "model.norm.weight": ["fsdp",],
            #     "lm_head.weight": ["fsdp", None],

            #     # mm_projector
            #     "model.mm_projector.*.weight": ["fsdp", None],
            #     "model.mm_projector.*.bias": ["fsdp",],
                
            #     # activations
            #     "model.layers.*[0]": ["fsdp", None, None],
            #     "lm_head": ["fsdp", None, None],
            # }

            # model = shard_torch_xla_model_from_config(model, config=sharding_config)

            # model = wrap_module(model, maybe_checkpoint)
            # model = wrap_module(model, maybe_add_barrier)
            # self.model = model
            #################################### ! CHANGES HERE ! ####################################
        else:
            self.model = model = FSDP(
                model,
                auto_wrap_policy=auto_wrap_policy,
                auto_wrapper_callable=auto_wrapper_callable,
                **fsdp_kwargs,
            )
        

        first_iter = True


        # Patch `xm.optimizer_step` should not reduce gradients in this case,
        # as FSDP does not need gradient reduction over sharded parameters.
        def patched_optimizer_step(optimizer, barrier=False, optimizer_args={}):
            loss = optimizer.step(**optimizer_args)
            
            if barrier:
                xm.mark_step()
            return loss

        xm.optimizer_step = patched_optimizer_step
        
    elif is_sagemaker_dp_enabled():
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[int(os.getenv("SMDATAPARALLEL_LOCAL_RANK"))]
        )
    elif self.args.parallel_mode == ParallelMode.DISTRIBUTED:
        if is_torch_neuroncore_available():
            return model
        kwargs = {}
        if self.args.ddp_find_unused_parameters is not None:
            kwargs["find_unused_parameters"] = self.args.ddp_find_unused_parameters
        elif isinstance(model, PreTrainedModel):
            # find_unused_parameters breaks checkpointing as per
            # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021
            kwargs["find_unused_parameters"] = not model.is_gradient_checkpointing
        else:
            kwargs["find_unused_parameters"] = True

        if self.args.ddp_bucket_cap_mb is not None:
            kwargs["bucket_cap_mb"] = self.args.ddp_bucket_cap_mb

        if self.args.ddp_broadcast_buffers is not None:
            kwargs["broadcast_buffers"] = self.args.ddp_broadcast_buffers

        self.accelerator.ddp_handler = DistributedDataParallelKwargs(**kwargs)

    # model.to(torch.bfloat16) # ! NOTE: this helps to reduce memory usage, but the impact on performance remains to be seen
    from scale_rae.utils import inspect_tensor_sharding
    print(model, flush=True)
    for name, param in model.named_parameters():
        print(name, inspect_tensor_sharding(param), flush=True)
    return model

Trainer._wrap_model = _wrap_model
################################################################################

################################################################################
# monkey-patched to transformers' qwen2 implementation
from torch_xla.experimental.custom_kernel import flash_attention
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.cache_utils import Cache
from scale_rae.utils import IS_XLA_AVAILABLE
import warnings
from typing import Optional, Tuple
import math

def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    logger.info_once("Calling monkey patch for Qwen2Attention.forward...")
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if not IS_XLA_AVAILABLE:
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

    else:
        
        # Check if model has stored attention bias for block vision loss mode
        attention_bias = getattr(self, '_current_attention_bias', None)
        attn_output = flash_attention(
            query_states, key_states, value_states, causal=True,
            ab=attention_bias,
            sm_scale=1. / math.sqrt(self.head_dim),
            partition_spec=('fsdp', None, None, None))

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

Qwen2Attention.forward = forward
################################################################################

################################################################################
# # monkey-patched to siglip's implementation
# from scale_rae.model.multimodal_encoder.llava_next_siglip_encoder import SigLipAttention, SigLipVisionTransformer
# def _siglip_attention_forward(
#     self,
#     hidden_states: torch.Tensor,
#     attention_mask: Optional[torch.Tensor] = None,
#     output_attentions: Optional[bool] = False,
# ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#     """Input shape: Batch x Time x Channel"""
#     logger.info_once("Calling monkey patch for SigLipAttention.forward...")

#     batch_size, q_len, _ = hidden_states.size()

#     query_states = self.q_proj(hidden_states)
#     key_states = self.k_proj(hidden_states)
#     value_states = self.v_proj(hidden_states)

#     query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#     key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#     value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

#     if not IS_XLA_AVAILABLE:
#         k_v_seq_len = key_states.shape[-2]
#         attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

#         if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
#             raise ValueError(f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is" f" {attn_weights.size()}")

#         if attention_mask is not None:
#             if attention_mask.size() != (batch_size, 1, q_len, k_v_seq_len):
#                 raise ValueError(f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}, but is {attention_mask.size()}")
#             attn_weights = attn_weights + attention_mask

#         # upcast attention to fp32
#         attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
#         attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
#         attn_output = torch.matmul(attn_weights, value_states)

#         if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
#             raise ValueError(f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is" f" {attn_output.size()}")
#     else:
#         attn_output = flash_attention(
#             query_states, key_states, value_states, causal=False,
#             ab=attention_mask,
#             sm_scale=self.scale,
#             partition_spec=('fsdp', None, None, None))
#         attn_weights = None

#     attn_output = attn_output.transpose(1, 2).contiguous()
#     attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

#     attn_output = self.out_proj(attn_output)

#     return attn_output, attn_weights

# def _siglip_vit_forward(
#     self,
#     pixel_values,
#     output_attentions: Optional[bool] = None,
#     output_hidden_states: Optional[bool] = None,
#     return_dict: Optional[bool] = None,
# ):
#     logger.info_once("Calling monkey patch for SigLipVisionTransformer.forward...")

#     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#     output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#     hidden_states = self.embeddings(pixel_values)

#     #################################### ! CHANGES HERE ! ####################################
#     hidden_states_padded = False
#     hidden_states_seqlen = None
#     attention_mask = None
#     if hidden_states.size(1) % 512 != 0:
#         hidden_states_padded = True
#         hidden_states_seqlen = hidden_states.size(1)
#         hidden_states = torch.cat([hidden_states, torch.zeros(hidden_states.size(0), 512 - hidden_states.size(1) % 512, hidden_states.size(2)).to(hidden_states.device)], dim=1)

#         if not hasattr(self, "cached_attention_mask"):
#             attention_mask = torch.zeros(hidden_states.size(0), self.config.num_attention_heads, hidden_states.size(1), hidden_states.size(1)).to(hidden_states.dtype)
#             attention_mask[..., hidden_states_seqlen:] = torch.finfo(attention_mask.dtype).min
#             attention_mask = attention_mask.to(hidden_states.device)
#             self.cached_attention_mask = attention_mask
#         else:
#             attention_mask = self.cached_attention_mask
#     #################################### ! CHANGES HERE ! ####################################

#     encoder_outputs = self.encoder(
#         inputs_embeds=hidden_states,
#         attention_mask=attention_mask, # ! NOTE: CHANGE HERE !
#         output_attentions=output_attentions,
#         output_hidden_states=output_hidden_states,
#         return_dict=return_dict,
#     )
    
#     #################################### ! CHANGES HERE ! ####################################
#     if hidden_states_padded:
#         # Change tuple to list to allow modification
#         encoder_outputs.hidden_states = list(encoder_outputs.hidden_states)
#         # Trunc the last hidden state to the original length
#         encoder_outputs.hidden_states[-1] = encoder_outputs.hidden_states[-1][:, :hidden_states_seqlen, :].clone()
#     #################################### ! CHANGES HERE ! ####################################

#     last_hidden_state = encoder_outputs[0]
#     last_hidden_state = self.post_layernorm(last_hidden_state)

#     pooled_output = self.head(last_hidden_state)

#     if not return_dict:
#         return (last_hidden_state, pooled_output) + encoder_outputs[1:]

#     from transformers.modeling_outputs import BaseModelOutputWithPooling
#     return BaseModelOutputWithPooling(
#         last_hidden_state=last_hidden_state,
#         pooler_output=pooled_output,
#         hidden_states=encoder_outputs.hidden_states,
#         attentions=encoder_outputs.attentions,
#     )

# SigLipAttention.forward = _siglip_attention_forward
# SigLipVisionTransformer.forward = _siglip_vit_forward
# # not use because it does not help to reduce memory usage or speed up
# # the above code is monkey-patched to siglip's implementation
# # siglip's seq len is 729, not divisible by 512 so we cannot use torchxla's flash attention here
################################################################################

from fsspec.core import url_to_fs
from torch_xla.experimental.distributed_checkpoint import CheckpointManager

################################################################################
# monkey-patched to ScaleRAETrainer for checkpoint management
from scale_rae.train.scale_rae_trainer import ScaleRAETrainer
from transformers import Trainer
import dataclasses
import json
import torch_xla.core.xla_model as xm

def __init__(self, *args, **kwargs):
    logger.info("Calling spmd monkey patch for ScaleRAETrainer.__init__...")
    super(ScaleRAETrainer, self).__init__(*args, **kwargs)
    logger.info("Finish super init for ScaleRAETrainer.__init__...")

    self.consolidate_counter = 0
    logger.info(f"Create checkpoint manager with output_dir {self.args.output_dir.replace('/mnt/', 'gs://')}...")

    dist.barrier()
    global _LOCAL_PROCESS_GROUP
    if dist.get_rank() == 0:
        logger.info("Create checkpoint folder if not exist on rank 0...")
        os.makedirs(self.args.output_dir.replace("gs://", "/mnt/"), exist_ok=True)
    dist.barrier()
    self.checkpoint_manager = CheckpointManager(
        path=self.args.output_dir.replace("gs://", "/mnt/"), # ! NOTE: hard code to gcsfuse
        save_interval=self.args.save_steps,
        max_to_keep=5, # ! NOTE: hard code to 10 but need to be configurable
        process_group=_LOCAL_PROCESS_GROUP,
    )
    dist.barrier()
    logger.info("Finish create checkpoint manager...")
    logger.info(f"Create fs with output_dir {self.args.output_dir.replace('gs://', '/mnt/')}")
    self.fs, _ = url_to_fs(self.args.output_dir.replace("gs://", "/mnt/"))
    dist.barrier()
    logger.info("Finish create fs...")

ScaleRAETrainer.__init__ = __init__

def get_train_dataloader(self):
    logger.info("Calling spmd monkey patch for ScaleRAETrainer.get_train_dataloader...")
    out = super(ScaleRAETrainer, self).get_train_dataloader()
    # Forward set_epoch to dataset if the dataloader itself doesn't expose it (IterableDataset shuffle)
    
    # try:
    #     dataset = getattr(out, "dataset", None)
    #     if dataset is not None and hasattr(dataset, "set_epoch") and not hasattr(out, "set_epoch"):
    #         def _forward_set_epoch(epoch):
    #             try:
    #                 dataset.set_epoch(epoch)
    #             except Exception as e:
    #                 logger.warning(f"Failed to forward set_epoch to dataset: {e}")
    #         setattr(out, "set_epoch", _forward_set_epoch)
    #         logger.info("Attached set_epoch forwarder to train dataloader (delegates to dataset.set_epoch)")
    # except Exception as e:
    #     logger.warning(f"Unable to attach set_epoch forwarder: {e}")

    return out

ScaleRAETrainer.get_train_dataloader = get_train_dataloader
from torch_xla.experimental.distributed_checkpoint import prime_optimizer
from safetensors.torch import save_file, load_file

def save_with_zstd(state_dict, path):
    """Saves a state_dict to a Zstandard-compressed file."""
    # Use a buffer to avoid writing an intermediate file to disk
    buffer = BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)  # Rewind the buffer to the beginning

    # Compress the buffer's content and write to the final file
    with open(path, 'wb') as f:
        f.write(zstd.compress(buffer.getvalue()))


def _save_checkpoint(self, model, trial, metrics=None):
    logger.info("Calling spmd monkey patch for ScaleRAETrainer._save_checkpoint...")
    
    import json
    import dataclasses

    # prime_optimizer(self.optimizer)          # <-- add this line
    xm.mark_step()


    # Save webdataset state per-rank (similar to RNG state)
    from scale_rae.train.webdataset_trainer import WebDatasetLazySupervisedDataset

    print("type(self.train_dataset):", type(self.train_dataset))
    print("isinstance(self.train_dataset, WebDatasetLazySupervisedDataset):", isinstance(self.train_dataset, WebDatasetLazySupervisedDataset))
    


    # print_optimizer_stats_local(self.optimizer, self.model)



    logger.info("Creating state dict...")
    state_dict = {
        "model": self.model.state_dict(),
        "optimizer": self.optimizer.state_dict(),
    }

    logger.info("Creating state dict done.")

    if self.checkpoint_manager.save(
        self.state.global_step, state_dict, force=True
    ):  # use save instead of save_async to make sure the checkpoint is saved
        checkpoint_dir = self.checkpoint_manager._get_path(self.state.global_step)
        logger.info(f"Saving checkpoint at {checkpoint_dir}")
    else:
        checkpoint_dir = None


    rank = dist.get_rank()
    if isinstance(self.train_dataset, WebDatasetLazySupervisedDataset):
        logger.info(f"Saving WebDataset state for rank {rank}...")
        try:
            # Aggregate per-worker states (optional flush by workers)
            import glob
            state_dir = os.getenv("WEBDS_STATE_DIR", os.path.join(os.getcwd(), ".webds_state"))
            aggregated_workers = {}

            # Merge main-process view (may be empty when using multi-workers)
            if isinstance(getattr(self.train_dataset, "resume_state", None), dict):
                aggregated_workers.update(self.train_dataset.resume_state)

            # Merge per-worker files if present
            pattern = os.path.join(state_dir, f"webds_rank{rank}_worker*.json")
            for path in glob.glob(pattern):
                try:
                    with open(path, "r") as wf:
                        loaded = json.load(wf)
                    workers = loaded.get("workers", loaded)
                    if isinstance(workers, dict):
                        aggregated_workers.update(workers)
                except Exception as e:
                    logger.warning(f"Failed to read worker state file {path}: {e}")

            if checkpoint_dir is not None:
                webdataset_state = {
                    "epoch": getattr(self.train_dataset, "epoch", 0),
                    "tar_shuffle_seed": 42 + getattr(self.train_dataset, "epoch", 0),
                    "workers": aggregated_workers,
                }
                webdataset_state_path = os.path.join(checkpoint_dir, f"webdataset_state_rank{rank}.json")
                with self.fs.open(webdataset_state_path, "w") as f:
                    f.write(json.dumps(webdataset_state, indent=2))
                logger.info(f"WebDataset state saved to {webdataset_state_path} (workers: {len(aggregated_workers)})")
            else:
                logger.warning("Checkpoint dir is None; skipping WebDataset state save")
        except Exception as e:
            logger.warning(f"Failed to save WebDataset resume state for rank {rank}: {e}")
    else:
        logger.info("No WebDataset state to save")


    # logger.info("Saving optimizer shards...")


    # rank       = dist.get_rank()        # 0 … 63   ← the host you're on
    # rank       = dist.get_rank()        # 0 … 63   ← the host you're on
    # world_size = dist.get_world_size()  # 64       ← total TPU‑VMs

    # WEIGHTS_NAME = ".pt.zst"
    # logger.info(f"Saving optimizer shards with rank {rank} and world_size {world_size}...")

    # SHARD_NAME_OPT = f'opt_rank-{rank:08d}-of-{world_size:08d}{WEIGHTS_NAME}'

    # SHARD_NAME_OPT_PATH = os.path.join(self.args.output_dir.replace("gs://", "/mnt/"), str(self.state.global_step), SHARD_NAME_OPT)


    # logger.info(f"SHARD_NAME_OPT_PATH: {SHARD_NAME_OPT_PATH}")

    # save_with_zstd(self.optimizer.state_dict(), SHARD_NAME_OPT_PATH)

    # logger.info("Saving optimizer shards done.")



    # logger.info("Saving checkpoint done.")
    # dist.barrier() # NOTE

    logger.info("Create random number generator state...")
    # save rng state
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torchxla": xm.get_rng_state(),
    }



    logger.info("Saving random number generator state...")
    with self.fs.open(os.path.join(checkpoint_dir, f"rng_state_rank{rank}.pth"), "wb") as f:
        torch.save(rng_state, f)
    logger.info("Saving random number generator state done.")
    

    logger.info("Saving lr_scheduler state and trainer state and model config...")
    if rank == 0:
        # save lr_scheduler state
        lr_state = self.lr_scheduler.state_dict()
        with self.fs.open(os.path.join(checkpoint_dir, "lr_scheduler.pth"), "wb") as f:
            torch.save(lr_state, f)
        logger.info("Saving lr_scheduler state done.")

        # save trainer state
        json_string = json.dumps(dataclasses.asdict(self.state), indent=2, sort_keys=True) + "\n"
        with self.fs.open(os.path.join(checkpoint_dir, "trainer_state.json"), "w") as f:
            f.write(json_string)
        logger.info("Saving trainer state done.")

        # save model config
        self.model.config.save_pretrained(checkpoint_dir.replace("gs://", "/mnt/"))
        logger.info("Saving model config done.")

    dist.barrier()

    # self.consolidate_counter += 1
    # if self.consolidate_counter % self.args.consolidate_interval == 0:
    #     self._save(checkpoint_dir)

    dist.barrier()
    xm.rendezvous("_save_checkpoint")




ScaleRAETrainer._save_checkpoint = _save_checkpoint

def _load_rng_state(self, resume_from_checkpoint):
    logger.info("Calling spmd monkey patch for ScaleRAETrainer._load_rng_state...")
    if resume_from_checkpoint is None:
        logger.info("resume_from_checkpoint is None, skip loading rng state")
        return
    resume_from_checkpoint = resume_from_checkpoint.replace("gs://", "/mnt/")
    logger.info(f"Loading RNG states from {resume_from_checkpoint}")

    with self.fs.open(os.path.join(resume_from_checkpoint, f"rng_state_rank{dist.get_rank()}.pth"), "rb") as f:
        rng_state = torch.load(f, map_location="cpu", weights_only=False)

    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"])
    xm.set_rng_state(rng_state["torchxla"])

    logger.info(f"Loading RNG states from {resume_from_checkpoint} successfully")
    xm.rendezvous("_load_rng_state")

def _load_optimizer_and_scheduler(self, resume_from_checkpoint):

    logger.info("Calling spmd monkey patch for ScaleRAETrainer._load_optimizer_and_scheduler...")
    if resume_from_checkpoint is None:
        logger.info("resume_from_checkpoint is None, skip loading optimizer and scheduler states")
        return
    resume_from_checkpoint = resume_from_checkpoint.replace("gs://", "/mnt/")
    logger.info(f"Loading optimizer and scheduler states from {resume_from_checkpoint}")

    with self.fs.open(os.path.join(resume_from_checkpoint, "lr_scheduler.pth"), "rb") as f:
        lr_state = torch.load(f, map_location="cpu", weights_only=False)
    self.lr_scheduler.load_state_dict(lr_state)

    logger.info(f"Loading optimizer and scheduler states from {resume_from_checkpoint} successfully")
    xm.rendezvous("_load_optimizer_and_scheduler")



# Ultra-simple diagnostic - just check if momentum exists in optimizer state
def check_optimizer_momentum_simple(optimizer):
    """Simple check: do we have momentum buffers?"""
    print("=== ULTRA-SIMPLE OPTIMIZER CHECK ===")
    
    # Get state_dict (this should have 441 entries based on your logs)
    state_dict = optimizer.state_dict()
    state_entries = state_dict.get('state', {})
    
    print(f"Total state entries: {len(state_entries)}")
    
    if len(state_entries) == 0:
        print("❌ NO OPTIMIZER STATE AT ALL!")
        return
    
    # Check first few entries to see what they contain
    sample_count = min(5, len(state_entries))
    sample_states = list(state_entries.values())[:sample_count]
    
    print(f"\nChecking {sample_count} sample parameter states:")
    
    for i, param_state in enumerate(sample_states):
        print(f"\n--- SAMPLE PARAMETER {i+1} ---")
        print(f"  Step: {param_state.get('step', 'MISSING')}")
        
        # Check exp_avg (first moment)
        if 'exp_avg' in param_state:
            exp_avg = param_state['exp_avg']
            norm = exp_avg.norm().item()
            is_zero = (exp_avg == 0).all().item()
            print(f"  ✅ exp_avg: norm={norm:.6f}, all_zero={is_zero}")
        else:
            print(f"  ❌ exp_avg: MISSING")
        
        # Check exp_avg_sq (second moment)  
        if 'exp_avg_sq' in param_state:
            exp_avg_sq = param_state['exp_avg_sq']
            norm = exp_avg_sq.norm().item()
            is_zero = (exp_avg_sq == 0).all().item()
            print(f"  ✅ exp_avg_sq: norm={norm:.6f}, all_zero={is_zero}")
        else:
            print(f"  ❌ exp_avg_sq: MISSING")
    
    # Quick summary
    has_momentum = any('exp_avg' in state for state in sample_states)
    momentum_is_zero = True
    
    if has_momentum:
        for state in sample_states:
            if 'exp_avg' in state:
                if not (state['exp_avg'] == 0).all().item():
                    momentum_is_zero = False
                    break
    
    print(f"\n=== DIAGNOSIS ===")
    if not has_momentum:
        print("❌ CRITICAL: No momentum buffers found!")
    elif momentum_is_zero:
        print("⚠️  WARNING: Momentum buffers exist but are all ZERO!")
    else:
        print("✅ GOOD: Non-zero momentum buffers found")
    
    return has_momentum and not momentum_is_zero


# Even simpler version to check specific parameter types
def check_diff_head_vs_regular_simple(optimizer, model):
    """Check if diff_head parameters have different optimizer state than regular ones"""
    print("\n=== DIFF_HEAD vs REGULAR COMPARISON ===")
    
    # Get a few parameter names from the model
    param_names = list(model.named_parameters())
    
    # Find examples of each type
    diff_head_example = None
    regular_example = None
    
    for name, param in param_names:
        if "diff_head" in name and diff_head_example is None:
            diff_head_example = (name, param)
        elif "diff_head" not in name and regular_example is None:
            regular_example = (name, param)
        
        if diff_head_example and regular_example:
            break
    
    if diff_head_example:
        print(f"📍 DIFF_HEAD example: {diff_head_example[0]}")
    else:
        print("❌ No diff_head parameters found!")
    
    if regular_example:
        print(f"📍 REGULAR example: {regular_example[0]}")
    else:
        print("❌ No regular parameters found!")
    
    # Just tell us what we found
    total_params = len(param_names)
    diff_head_count = sum(1 for name, _ in param_names if "diff_head" in name)
    regular_count = total_params - diff_head_count
    
    print(f"📊 Parameter counts:")
    print(f"  Total: {total_params}")
    print(f"  diff_head: {diff_head_count}")
    print(f"  regular: {regular_count}")

# ────────────────────────────────────────────────────────────────
# 1. tiny helper  — prints how many zero‑denominator slots exist
# ────────────────────────────────────────────────────────────────
def _debug_adam_state(tag, optimizer, model):
    """
    Prints learning rates and optimizer state for all diffusion‐head parameters.
    Call after optimizer.load_state_dict(...) to verify exp_avg, exp_avg_sq, and step.
    """
    print(f"\n===== DEBUG ADAM STATE: {tag} =====")
    # Print each param group's LR
    for i, group in enumerate(optimizer.param_groups):
        lr = group.get("lr", None)
        name = group.get("name", f"group_{i}")
        print(f"  Param group {i} ({name}): lr = {lr}")
    print("-" * 40)

    # Inspect all diffusion‐head parameters
    for name, param in model.named_parameters():
        if "diff_head" in name:
            print(f"Parameter: {name}")
            state = optimizer.state.get(param, None)
            if state is None:
                print("  --> No optimizer state for this parameter!")
                continue

            exp_avg    = state.get("exp_avg", None)
            exp_avg_sq = state.get("exp_avg_sq", None)
            step       = state.get("step", None)

            # if exp_avg is not None:
            #     print(f"  exp_avg    dtype = {exp_avg.dtype}")
            #     if exp_avg.dtype != torch.float32:
            #         print(f"  --> Converting exp_avg from {exp_avg.dtype} to float32")
            #         state["exp_avg"] = exp_avg.to(torch.float32)

            # if exp_avg_sq is not None:
            #     print(f"  exp_avg_sq dtype = {exp_avg_sq.dtype}")
            #     if exp_avg_sq.dtype != torch.float32:
            #         print(f"  --> Converting exp_avg_sq from {exp_avg_sq.dtype} to float32")
            #         state["exp_avg_sq"] = exp_avg_sq.to(torch.float32)

            if exp_avg is not None:
                print(f"  exp_avg    max_abs = {exp_avg.abs().max().item():.6e}")
            else:
                print("  exp_avg    = None")

            if exp_avg_sq is not None:
                print(f"  exp_avg_sq max_abs = {exp_avg_sq.abs().max().item():.6e}")
            else:
                print("  exp_avg_sq = None")

            print(f"  step       = {step}")
    print("=" * 40 + "\n")


def _check_step_mismatch(opt, model, max_show=8):
    """
    Print parameters whose Adam `step` counter is 0 or very small
    while their 2‑moment is already non‑zero.
    """
    suspect = []
    state = opt.state_dict()["state"]
    for pid,s in state.items():
        v = s.get("exp_avg_sq", None)
        st = s.get("step", None)
        if isinstance(v, torch.Tensor) and st is not None:
            if st < 10 and torch.any(v > 0):   # v has data, but step is tiny
                suspect.append(pid)
                if len(suspect) >= max_show:
                    break
    # map pid → name
    id2name = {id(p): n for n,p in model.named_parameters()}
    names = [id2name.get(pid, f"<id:{pid}>") for pid in suspect]
    if names:
        print(f"[DBG] 🚨 step mismatch on {len(suspect)} params:", ", ".join(names))
    else:
        print("[DBG] ✔ all Adam `step` counters look OK")


# ── TPU‑SAFE DIFF‑HEAD SLOT CHECK ─────────────────────────────────────────────
# place   right after:  optimizer.load_state_dict(...)
#         inside your  _load_from_checkpoint(...)
import torch, torch.distributed as dist, torch_xla.core.xla_model as xm

# def debug_and_patch_diff_head_slots(trainer, resume_step: int, patch: bool = True):
#     """
#     • Detects diffusion‑head parameters that either
#         – have **no** optimizer state on this rank, or
#         – have a local `step` counter ≠ resume_step.
#     • If `patch=True` (default) creates/fixes the slot in‑place so every
#       rank starts with consistent Adam statistics.
#     """
#     opt   = trainer.optimizer
#     model = trainer.model
#     local_issues = []                         # (name, msg)

#     for name, p in model.named_parameters():
#         if "diff_head" not in name or not p.requires_grad:
#             continue

#         st = opt.state.get(p, None)
#         if st is None:
#             local_issues.append((name, "NO_STATE"))
#             if patch:          # ────── create zero slot ──────
#                 opt.state[p] = {
#                     "step":       resume_step,
#                     "exp_avg":    torch.zeros_like(p.data, dtype=p.data.dtype),
#                     "exp_avg_sq": torch.zeros_like(p.data, dtype=p.data.dtype),
#                 }
#         else:
#             step_here = int(st.get("step", 0))
#             if step_here != resume_step:
#                 local_issues.append((name, f"step={step_here}"))
#                 if patch:      # ────── fix step counter ──────
#                     st["step"] = resume_step

#     # ‑‑‑ gather diagnostics to rank‑0 ‑‑‑
#     world = dist.get_world_size()
#     gathered = [None] * world
#     dist.all_gather_object(gathered, local_issues)          # official API :contentReference[oaicite:0]{index=0}
#     xm.mark_step()       # flush lazy ops so prints show up promptly

#     if dist.get_rank() == 0:
#         print("\n===== DIFF‑HEAD SLOT CHECK =====")
#         any_issue = False
#         for r, issues in enumerate(gathered):
#             for n, msg in issues:
#                 print(f"R{r:02d}  {msg:<10} {n}")
#                 any_issue = True
#         if not any_issue:
#             print("All diffusion‑head slots present and step counters match.")
#         print("================================\n")

def print_optimizer_stats_local(optimizer, model):
    """
    A focused and robust test to count ONLY the negative values in the
    optimizer's 2nd moment buffer (exp_avg_sq) on each TPU rank locally.
    """
    rank = xm.get_ordinal()

    # Each rank will count its local negative values
    local_llm_neg_count = 0
    local_diff_head_neg_count = 0

    for name, p in model.named_parameters():
        if p.requires_grad:
            state = optimizer.state.get(p)
            if state and 'exp_avg_sq' in state:
                exp_avg_sq_tensor = state['exp_avg_sq'].detach()
                
                # Directly count values less than zero on the local tensor shard
                neg_count = (exp_avg_sq_tensor < 0).sum().item()
                
                if "diff_head" in name:
                    local_diff_head_neg_count += neg_count
                else:
                    local_llm_neg_count += neg_count
    
    # --- Print the results for this rank ---
    print("\n" + "="*80, flush=True)
    print(f"[Rank {rank:03d}] --- Negative Value Count in `exp_avg_sq` ---", flush=True)
    print(f"  [Rank {rank:03d}] LLM Params Negative Count:       {local_llm_neg_count:,}", flush=True)
    print(f"  [Rank {rank:03d}] Diff_Head Params Negative Count: {local_diff_head_neg_count:,}", flush=True)
    print("="*80 + "\n", flush=True)

    dist.barrier()
import torch
from torch.utils._pytree import tree_map
import torch_xla
import torch_xla.core.xla_model as xm

def monkey_patch_prime_optimizer(optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
  """
  Prime the optimizer state by running a dummy weight update.

  Optimizer state isn't created until after the first training step. Since the
  distributed checkpointing library loads the state_dict in-place, the
  optimizer state must already exist before loading the checkpoint.

  This utility method runs a dummy weight update with zero gradient to ensure
  the optimizer state exists and can be loaded into.

  **Warning** This method calls `optimizer.step`, which can impact the
  optimizer's state and model parameters. Therefore, it should only be used
  prior to restoring a checkpoint, when the state and parameters will be
  immediately overwritten.

  Args:
    optimizer: The optimizer whose state should be primed for checkpoint
               loading.
  """

  # Initial `torch_xla.sync()` to ensure all param_groups are backed by device
  # data.
  torch_xla.sync()
  xm.wait_device_ops()

#   print(f"[DEBUG] optimizer.param_groups {optimizer.param_groups}")

  def zero_grad(x):
    if isinstance(x, torch.Tensor) and x.requires_grad:
      x.grad = torch.zeros_like(x, requires_grad=False)
      param_sharding = torch_xla._XLAC._get_xla_op_sharding(x)
      if param_sharding:
        # Match the gradient sharding to the parameter's.
        torch_xla._XLAC._xla_mark_sharding(x.grad, param_sharding)

  tree_map(zero_grad, optimizer.param_groups)
  optimizer.step()
  torch_xla.sync()
  xm.wait_device_ops()
  return optimizer


def prime_optimizer_manually(optimizer):
    """
    Initializes the optimizer state buffers manually without calling
    optimizer.step(), to avoid compiling a premature XLA graph. This
    is the safe alternative to the original prime_optimizer.
    """
    logger.warning("<<<< Applying new experiment: Manually priming optimizer state >>>>")
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.requires_grad:
                if p not in optimizer.state:
                    # Create the state dictionary for the parameter
                    optimizer.state[p] = {}
                    state = optimizer.state[p]
                    # AdamW state initialization
                    state['step'] = torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
    logger.warning("<<<< Manual priming complete >>>>")
    optimizer.step()

    xm.mark_step()

    
# from torch_xla._XLAC import _get_xla_op_sharding, _xla_mark_sharding

def safe_prime_optimizer(optim, resume_step: int):
    """
    Allocates *empty* AdamW slots on the correct XLA device/shard
    so that CheckpointManager can load shards in‑place.
    • never touches model weights
    • does NOT run optimizer.step()
    • initialises step counter to `resume_step`
    """
    for group in optim.param_groups:
        for p in group["params"]:
            if not p.requires_grad or p in optim.state:
                continue

            sharding = torch_xla._XLAC._get_xla_op_sharding(p)
            print("sharding:", sharding)
            exp_avg    = torch.zeros_like(p, dtype=torch.float32)
            exp_avg_sq = torch.zeros_like(p, dtype=torch.float32)

            # keep sharding identical to the parameter
            # if sharding:
            #     torch_xla._XLAC._xla_mark_sharding(exp_avg,    sharding)
            #     torch_xla._XLAC._xla_mark_sharding(exp_avg_sq, sharding)

            optim.state[p] = {
                "step":       torch.tensor(0, device=p.device, dtype=torch.int64),
                "exp_avg":    exp_avg,
                "exp_avg_sq": exp_avg_sq,
            }
    xm.mark_step()


def _patch_missing_adam_slots(model, optimizer, resume_step):
    import torch_xla.core.xla_model as xm
    created = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        st = optimizer.state.get(p)
        if st is None:                                # ─ missing slot
            optimizer.state[p] = {
                "step":       resume_step,
                "exp_avg":    torch.zeros_like(p, dtype=torch.float32),
                "exp_avg_sq": torch.zeros_like(p, dtype=torch.float32),
            }
            created += 1
            continue

        if st.get("step", 0) != resume_step:          # ─ wrong step
            st["step"] = resume_step

        for key in ("exp_avg", "exp_avg_sq"):         # ─ dtype + device
            if key not in st:
                continue
            t = st[key]
            if not xm.is_xla_tensor(t):               # ← **doc‑backed test**
                st[key] = t.to(p.device)
            if st[key].dtype != torch.float32:
                st[key] = st[key].to(torch.float32)
    xm.mark_step()
    return created



# def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
#     logger.info("Calling spmd monkey patch for ScaleRAETrainer._load_from_checkpoint...")
#     if resume_from_checkpoint is None:
#         logger.info("resume_from_checkpoint is None, skip loading model checkpoint")
#         return

#     resume_from_checkpoint = resume_from_checkpoint.replace("gs://", "/mnt/")
#     logger.info(f"Loading model checkpoint from {resume_from_checkpoint}")

#     # 1. Prime the optimizer (creates all slots)
#     from torch_xla.experimental.distributed_checkpoint import prime_optimizer
    
#     prime_optimizer(self.optimizer)
#     # prime_optimizer_manually(self.optimizer)


#     # monkey_patch_prime_optimizer(self.optimizer)
#     # safe_prime_optimizer(self.optimizer, resume_step=0)

#     # Debug after priming (optional)
#     # _debug_adam_state("AFTER prime_optimizer", self.optimizer, self.model)

#     # 2. Restore checkpoint (both model and optimizer state)
#     resume_step = int(resume_from_checkpoint.rstrip('/').split('/')[-1])
#     state_dict = {
#         "model": self.model.state_dict(),
#         "optimizer": self.optimizer.state_dict(),
#     }
#     self.checkpoint_manager.restore(resume_step, state_dict)

#     # 3. Load model weights
#     self.model.load_state_dict(state_dict["model"])

#     # 4. Load optimizer state
#     self.optimizer.load_state_dict(state_dict["optimizer"])

#     xm.rendezvous("_load_from_checkpoint")

def _one_time_load_and_step(optimizer_self, *args, **kwargs):
    """
    A one-time use wrapper for optimizer.step() that loads the checkpoint
    state just after the optimizer has been naturally initialized by the first
    call to step(). This is the core of the "Just-in-Time" loading fix.
    """
    logger.warning("<<<< Intercepted first optimizer.step(). Loading deferred state now... >>>>")
    
    # The live optimizer has just initialized its own state buffers naturally.
    # Now, we load our saved state from the checkpoint into those fresh buffers.

    # optimizer_self.load_state_dict(optimizer_self._state_to_load_on_first_step)
    
    # Clean up the temporary attribute and restore the original step method.
    # This ensures the hijack only ever runs once.
    del optimizer_self._state_to_load_on_first_step
    optimizer_self.step = optimizer_self._original_step
    
    logger.warning("<<<< State loaded and hijack removed. Calling original optimizer.step()... >>>>")
    
    # Call the original step method to perform the first update.
    return optimizer_self.step(*args, **kwargs)


# =================================================================================
# FINAL _load_from_checkpoint IMPLEMENTATION
# =================================================================================

def load_with_zstd(path, map_location='cpu'):
    """Loads a state_dict from a Zstandard-compressed file."""
    with open(path, 'rb') as f:
        compressed_data = f.read()
    
    decompressed_data = zstd.decompress(compressed_data)
    buffer = BytesIO(decompressed_data)
    return torch.load(buffer, map_location=map_location)


def _one_time_load_and_step(optimizer_self, *args, **kwargs):
    """
    Hijacks the first optimizer.step() call to load the optimizer state
    from a CONSOLIDATED checkpoint file, then restores the original step method.
    """
    logger.warning("<<<< Intercepted first optimizer.step(). Loading state from consolidated file... >>>>")
    
    # Load the consolidated optimizer state dict from the path we saved.
    # map_location='cpu' is a safeguard to prevent memory spikes on the TPU device.
    consolidated_optim_state = torch.load(optimizer_self._consolidated_optim_path, map_location="cpu")
    
    # The optimizer has just been naturally initialized. Now, load the state.
    optimizer_self.load_state_dict(consolidated_optim_state)
    logger.info(f"Successfully loaded optimizer state from {optimizer_self._consolidated_optim_path}")
    
    # Clean up and restore the original step method.
    del optimizer_self._consolidated_optim_path
    optimizer_self.step = optimizer_self._original_step
    
    logger.warning("<<<< State loaded and hijack removed. Calling original optimizer.step()... >>>>")
    
    # Call the now-restored, original step method to perform the first update.
    return optimizer_self.step(*args, **kwargs)


from torch.optim import Optimizer
from functools import partial


class LeanForensicStepWrapper:
    """
    A lightweight, targeted forensic tool. It checks for NaN/Inf values
    in 'diff_head' parameters and prints a confirmation if the check passes.
    """
    def __init__(self, optimizer, original_step_method, trainer):
        self.optimizer = optimizer
        self.original_step_method = original_step_method
        self.trainer = trainer
        self.params_to_inspect = self._get_diff_head_params()

        if not self.params_to_inspect and dist.get_rank() == 0:
            logger.warning("FORENSICS: Could not find any 'diff_head' parameters to inspect.")

    def _get_diff_head_params(self):
        # Create a reverse mapping from parameter object ID to its name
        id_to_name = {id(p): name for name, p in self.trainer.model.named_parameters()}
        
        diff_head_params = []
        for group in self.optimizer.param_groups:
            for param in group['params']:
                param_name = id_to_name.get(id(param), "")
                if "diff_head" in param_name:
                    diff_head_params.append((param_name, param))
        return diff_head_params

    def _check_tensor_health(self, title, tensor):
        # This is the updated lightweight check with confirmation logging.
        rank = dist.get_rank()
        if not isinstance(tensor, torch.Tensor):
            print(f"    [Rank {rank:03d}]     INFO: {title} is not a tensor.", flush=True)
            return

        # Perform the fast .any() checks
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        
        if has_nan or has_inf:
            # Loud, clear alert on error
            if has_nan:
                print(f"    [Rank {rank:03d}] !!! FOUND NaN in {title} !!!", flush=True)
            if has_inf:
                print(f"    [Rank {rank:03d}] !!! FOUND Inf in {title} !!!", flush=True)
        else:
            # Quiet confirmation that the check passed
            print(f"    [Rank {rank:03d}]     OK: {title} is clean.", flush=True)

    def step(self, *args, **kwargs):
        rank = dist.get_rank()
        if rank == 0:
            print("\n" + "="*80, flush=True)
            logger.warning("<<<< FORENSIC WRAPPER: Intercepting the failing step... >>>>")
        
        # --- 1. CHECK INPUTS ---
        if rank == 0: print(f"### FORENSICS: STATE *BEFORE* optimizer.step() ###", flush=True)
        for name, param in self.params_to_inspect:
            if rank == 0: print(f"--- Inspecting Parameter: {name} ---", flush=True)
            self._check_tensor_health(f"'Weight'", param.data)
            self._check_tensor_health(f"'Gradient'", param.grad)
            if param in self.optimizer.state:
                state = self.optimizer.state[param]
                self._check_tensor_health(f"'exp_avg'", state.get('exp_avg'))
                self._check_tensor_health(f"'exp_avg_sq'", state.get('exp_avg_sq'))
        dist.barrier()

        # --- 2. EXECUTE THE STEP ---
        if rank == 0: logger.warning("... executing original optimizer.step() now ...")
        result = self.original_step_method(*args, **kwargs)
        if rank == 0: logger.warning("... optimizer.step() finished.")
        dist.barrier()
        
        # --- 3. CHECK OUTPUTS ---
        if rank == 0: print(f"\n### FORENSICS: STATE *AFTER* optimizer.step() ###", flush=True)
        for name, param in self.params_to_inspect:
            if rank == 0: print(f"--- Inspecting Parameter: {name} ---", flush=True)
            self._check_tensor_health(f"'Updated Weight'", param.data)
            if param in self.optimizer.state:
                state = self.optimizer.state[param]
                self._check_tensor_health(f"'Updated exp_avg'", state.get('exp_avg'))
                self._check_tensor_health(f"'Updated exp_avg_sq'", state.get('exp_avg_sq'))
        dist.barrier()

        # --- 4. RESTORE AND FINISH ---
        self.optimizer.step = self.original_step_method
        if rank == 0:
            logger.warning("<<<< FORENSIC WRAPPER: Hijack removed. >>>>")
            print("="*80 + "\n", flush=True)
            
        return result

class UnifiedResumeHijacker:
    # This class is identical to the previous version. Its logic is sound.
    # It just hands off to the forensic wrapper if DEBUG_PRINT is True.
    def __init__(self, optimizer, trainer, strategy, resume_step=None, warmup_steps=5):
        self.optimizer = optimizer
        self.trainer = trainer
        self.strategy = strategy
        self.resume_step = resume_step
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.original_step_method = optimizer.step
        if self.strategy == "LOAD" and self.resume_step is None:
            raise ValueError("The 'LOAD' strategy requires a valid `resume_step`.")

    def step(self, *args, **kwargs):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            if dist.get_rank() == 0: logger.warning(f"<<<< [{self.strategy}] Warm-up: Step {self.current_step}/{self.warmup_steps}. Applying temporary zero LR/WD. >>>>")
            current_lrs = [group['lr'] for group in self.optimizer.param_groups]
            current_wds = [group.get('weight_decay', 0.0) for group in self.optimizer.param_groups]
            for group in self.optimizer.param_groups:
                group['lr'] = 0.0
                group['weight_decay'] = 0.0
            try:
                result = self.original_step_method(*args, **kwargs)
            finally:
                for i, group in enumerate(self.optimizer.param_groups):
                    group['lr'] = current_lrs[i]
                    group['weight_decay'] = current_wds[i]
            return result
        else:
            if dist.get_rank() == 0: logger.warning(f"<<<< [{self.strategy}] Warm-up complete. Executing strategy... >>>>")
            
            if self.strategy == "LOAD":
                state_to_populate = {"optimizer": self.optimizer.state_dict()}
                self.trainer.checkpoint_manager.restore(self.resume_step, state_to_populate)
                self.optimizer.load_state_dict(state_to_populate["optimizer"])
                if dist.get_rank() == 0: logger.info("<<<< [LOAD] Optimizer state successfully loaded. >>>>")
            else:
                if dist.get_rank() == 0: logger.info("<<<< [REBUILD] Optimizer has built fresh momentum. >>>>")
            
            DEBUG_PRINT = True

            if DEBUG_PRINT:
                if dist.get_rank() == 0: logger.warning("Handing off to LeanForensicStepWrapper for one step.")
                forensic_wrapper = LeanForensicStepWrapper(self.optimizer, self.original_step_method, self.trainer)
                self.optimizer.step = forensic_wrapper.step
            else:
                self.optimizer.step = self.original_step_method

            return self.optimizer.step(*args, **kwargs)

# The _load_from_checkpoint function remains unchanged.
# def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
#     if resume_from_checkpoint is None: return
#     resume_from_checkpoint_path = resume_from_checkpoint.replace("gs://", "/mnt/")
#     if dist.get_rank() == 0: logger.info(f"Loading checkpoint from {resume_from_checkpoint_path}")
#     resume_step = int(resume_from_checkpoint_path.rstrip('/').split('/')[-1])
#     model_state_to_load = {"model": self.model.state_dict()}
#     self.checkpoint_manager.restore(resume_step, model_state_to_load)
#     self.model.load_state_dict(model_state_to_load["model"])
#     if dist.get_rank() == 0: logger.info("Model weights loaded successfully.")
    
#     resume_strategy = os.getenv('RESUME_STRATEGY', 'LOAD').upper()
#     if resume_strategy not in ["LOAD", "REBUILD"]:
#         resume_strategy = "LOAD"
    
#     warmup_steps = 5 if resume_strategy == "LOAD" else 20 # Using 20 for rebuild based on our analysis
#     hijacker = UnifiedResumeHijacker(self.optimizer, self, strategy=resume_strategy, resume_step=resume_step, warmup_steps=warmup_steps)
#     self.optimizer.step = hijacker.step
    
#     if dist.get_rank() == 0: logger.warning(f"Applying '{resume_strategy}' strategy. Optimizer hijacked for a warm-up of {hijacker.warmup_steps} steps.")
#     xm.rendezvous("_load_from_checkpoint")



def _load_from_checkpoint(self, resume_from_checkpoint, model_only=False):
    logger.info("Calling spmd monkey patch for ScaleRAETrainer._load_from_checkpoint...")
    if resume_from_checkpoint is None:
        logger.info("resume_from_checkpoint is None, skip loading model checkpoint")
        return
    resume_from_checkpoint = resume_from_checkpoint.replace("gs://", "/mnt/")
    logger.info(f"Loading model checkpoint from {resume_from_checkpoint}")

    # Before restoring the checkpoint, the optimizer state must be primed
    # to allow state to be loaded into it.
    if not model_only:
        from torch_xla.experimental.distributed_checkpoint import prime_optimizer
        prime_optimizer(self.optimizer)

    resume_step = int(resume_from_checkpoint.rstrip('/').split('/')[-1])

    if not model_only:
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
    else:
        state_dict = {"model": self.model.state_dict()}

    self.checkpoint_manager.restore(resume_step, state_dict)

    self.model.load_state_dict(state_dict["model"])
    if not model_only:
        self.optimizer.load_state_dict(state_dict["optimizer"])

    logger.info(f"Loaded checkpoint from {resume_from_checkpoint} successfully")
    
    # Load webdataset state per-rank (similar to RNG state)
    # if not model_only:
    from scale_rae.train.webdataset_trainer import WebDatasetLazySupervisedDataset
    if isinstance(self.train_dataset, WebDatasetLazySupervisedDataset):
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        logger.info(f"Loading WebDataset resume state for rank {rank}...")
        # Each rank loads its own state file (no TPU communication needed)
        webdataset_state_path = os.path.join(resume_from_checkpoint, f"webdataset_state_rank{rank}.json") 
        self.train_dataset.set_resume_checkpoint_file(webdataset_state_path)
        logger.info("WebDataset resume state loaded successfully")
    else:
        logger.info("WebDataset resume state not loaded because train_dataset is not a WebDatasetLazySupervisedDataset")


    xm.rendezvous("_load_from_checkpoint")

        # 🚨 RESCUE MODE: If rescue_ckpt is enabled, save and exit here!
    if getattr(self.args, 'rescue_ckpt', False):
        logger.info("🚨 RESCUE MODE: Checkpoint loaded successfully, now performing rescue save...")
        self._rescue_save_sharded()
        logger.info("🚨 RESCUE COMPLETE: Model saved successfully, exiting...")
        import sys
        sys.exit(0)  # Exit cleanly after rescue
    
    

ScaleRAETrainer._load_rng_state = _load_rng_state
ScaleRAETrainer._load_optimizer_and_scheduler = _load_optimizer_and_scheduler
ScaleRAETrainer._load_from_checkpoint = _load_from_checkpoint

def sync_to_cpu(state_dict):
    def convert_fn(item):
        if isinstance(item, torch.Tensor):
            item = xm._maybe_convert_to_cpu(item).to(torch.float32)
            return item
        elif isinstance(item, dict):
            return {k: convert_fn(v) for k,v in item.items()}
        elif isinstance(item, list):
            return [convert_fn(v) for v in item]
        elif isinstance(item, tuple):
            return tuple(convert_fn(v) for v in item)
        else:
            return item
    state_dict = {
        k: convert_fn(v) for k,v in state_dict.items()
    }
    return state_dict

def sync_to_cpu_safe(state_dict):
    """
    Safe manual CPU sync that avoids large multi-tensor sync to reduce OOM
    and ensures computations are flushed to prevent NaNs.
    """
    # Ensure all pending XLA ops are executed before reading tensors
    xm.mark_step()
    xm.wait_device_ops()

    nan_params = []  # Track parameters with NaN values
    total_params = 0
    
    def convert_fn(item, param_name=""):
        nonlocal total_params
        if isinstance(item, torch.Tensor):
            total_params += 1
            # Detach and copy to host in float32 to avoid reading from device buffers
            cpu_tensor = item.detach().to(dtype=torch.float32, device='cpu', copy=True)
            
            # Check for NaN values
            if torch.isnan(cpu_tensor).any():
                nan_count = torch.isnan(cpu_tensor).sum().item()
                total_elements = cpu_tensor.numel()
                nan_params.append({
                    'name': param_name,
                    'shape': tuple(cpu_tensor.shape),
                    'nan_count': nan_count,
                    'total_elements': total_elements,
                    'nan_ratio': nan_count / total_elements
                })
                logger.warning(f"🚨 NaN detected in parameter '{param_name}': {nan_count}/{total_elements} elements ({nan_count/total_elements:.2%})")
            
            return cpu_tensor
        elif isinstance(item, dict):
            return {k: convert_fn(v, f"{param_name}.{k}" if param_name else k) for k, v in item.items()}
        elif isinstance(item, list):
            return [convert_fn(v, f"{param_name}[{i}]" if param_name else f"[{i}]") for i, v in enumerate(item)]
        elif isinstance(item, tuple):
            return tuple(convert_fn(v, f"{param_name}({i})" if param_name else f"({i})") for i, v in enumerate(item))
        else:
            return item

    result = {k: convert_fn(v, k) for k, v in state_dict.items()}
    
    # Summary report
    if nan_params:
        logger.error(f"🚨 NaN DETECTION SUMMARY: Found NaN values in {len(nan_params)} out of {total_params} parameters!")
        for param_info in nan_params:
            logger.error(f"  - {param_info['name']}: {param_info['nan_count']}/{param_info['total_elements']} NaN ({param_info['nan_ratio']:.2%}), shape={param_info['shape']}")
        logger.error("⚠️  WARNING: Saving model with NaN parameters may cause issues during loading/inference!")
    else:
        logger.info(f"✅ NaN check passed: All {total_params} parameters are NaN-free")
    
    return result

def _save(self, output_dir, state_dict=None):
    logger.info("Calling spmd monkey patch for ScaleRAETrainer._save...")
    output_dir = output_dir.replace("gs://", "/mnt/")
    logger.info(f"Saving model checkpoint to {output_dir}")

    state_dict = {"model": self.model.state_dict()}
    # Use manual CPU sync instead of problematic sync_to_cpu
    def manual_sync_to_cpu(state_dict):
        logger.info("Converting state dict to CPU using manual method...")
        def convert_fn(item):
            if isinstance(item, torch.Tensor):
                # Force XLA mark step before conversion
                xm.mark_step()
                # Manual CPU conversion
                item = item.detach().cpu().to(torch.float32)
                return item
            elif isinstance(item, dict):
                return {k: convert_fn(v) for k,v in item.items()}
            elif isinstance(item, list):
                return [convert_fn(v) for v in item]
            elif isinstance(item, tuple):
                return tuple(convert_fn(v) for v in item)
            else:
                return item
        state_dict = {
            k: convert_fn(v) for k,v in state_dict.items()
        }
        return state_dict
    

    state_dict = sync_to_cpu_safe(state_dict)
    logger.info("Safe manual CPU sync completed successfully")

    # try:
    #     state_dict = sync_to_cpu(state_dict)
    #     logger.info("XLA sync_to_cpu completed successfully")
    # except Exception as e:
    #     logger.warning(f"sync_to_cpu failed with {e}. Falling back to safe manual CPU sync...")
    #     state_dict = sync_to_cpu_safe(state_dict)
    #     logger.info("Safe manual CPU sync completed successfully")

    if dist.get_rank() == 0:
        logger.info("Saving model state dict to CPU")
        if output_dir.startswith("/mnt/") and not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with BytesIO() as buffer, self.fs.open(os.path.join(output_dir, "model.pth.zstd"), 'wb') as f:
            torch.save(state_dict, buffer)
            f.write(zstd.compress(buffer.getvalue()))
        logger.info("Saving model state dict to CPU successfully")
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir.replace("gs://", "/mnt/"))
        logger.info("Saving tokenizer to CPU successfully")
        with self.fs.open(os.path.join(output_dir, "training_args.bin"), 'wb') as f:
            torch.save(self.args, f)
        logger.info("Saving training args to CPU successfully")
        # ! NOTE: make sure model config is the last one to be saved
        self.model.config.save_pretrained(output_dir.replace("gs://", "/mnt/"))
        logger.info("Saving model config to CPU successfully")
    dist.barrier()

    logger.info(f"Saved model checkpoint to {output_dir} successfully")
    xm.rendezvous("_save")

ScaleRAETrainer._save = _save

def _rescue_save_sharded(self, output_dir=None, state_dict=None):
    """
    Emergency rescue save: Follow exact _save pattern but with manual CPU sync
    This avoids the memory bottleneck that causes hangs on large TPU pods
    """
    logger.info("🚨 Calling rescue save with manual CPU sync...")
    
    # Use provided output_dir or create rescue variant
    if output_dir is None:
        output_dir = self.args.output_dir.replace("gs://", "/mnt/") + "-rescue"
    else:
        output_dir = output_dir.replace("gs://", "/mnt/")
    
    logger.info(f"🚨 RESCUE: Saving model checkpoint to {output_dir}")

    # Step 1: Create state dict (same as original)
    state_dict = {"model": self.model.state_dict()}
    
    # Step 2: Manual CPU sync instead of sync_to_cpu(state_dict)
    logger.info("🚨 RESCUE: Manual CPU sync starting...")
    def manual_sync_to_cpu(state_dict):
        logger.info("Converting state dict to CPU using manual method...")
        def convert_fn(item):
            if isinstance(item, torch.Tensor):
                # Force XLA mark step before conversion
                xm.mark_step()
                # Manual CPU conversion
                item = item.detach().cpu().to(torch.float32)
                return item
            elif isinstance(item, dict):
                return {k: convert_fn(v) for k,v in item.items()}
            elif isinstance(item, list):
                return [convert_fn(v) for v in item]
            elif isinstance(item, tuple):
                return tuple(convert_fn(v) for v in item)
            else:
                return item
        state_dict = {
            k: convert_fn(v) for k,v in state_dict.items()
        }
        return state_dict
    
    state_dict = sync_to_cpu_safe(state_dict)
    logger.info("🚨 RESCUE: Safe manual CPU sync completed")

    # Step 3: Save (same as original, but only rank 0)
    if dist.get_rank() == 0:
        logger.info("🚨 RESCUE: Saving model state dict to CPU")
        if output_dir.startswith("/mnt/") and not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Save compressed model (same as original)
        with BytesIO() as buffer, self.fs.open(os.path.join(output_dir, "model.pth.zstd"), 'wb') as f:
            torch.save(state_dict, buffer)
            f.write(zstd.compress(buffer.getvalue()))
        logger.info("🚨 RESCUE: Saving model state dict to CPU successfully")
        
        # Save tokenizer (same as original)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir.replace("gs://", "/mnt/"))
        logger.info("🚨 RESCUE: Saving tokenizer to CPU successfully")
        
        # Save training args (same as original)
        with self.fs.open(os.path.join(output_dir, "training_args.bin"), 'wb') as f:
            torch.save(self.args, f)
        logger.info("🚨 RESCUE: Saving training args to CPU successfully")
        
        # Save model config (same as original)
        self.model.config.save_pretrained(output_dir.replace("gs://", "/mnt/"))
        logger.info("🚨 RESCUE: Saving model config to CPU successfully")
        
        # Save rescue metadata
        rescue_metadata = {
            "rescue_info": {
                "original_run": getattr(self.args, 'run_name', 'unknown'),
                "global_step": self.state.global_step,
                "rescue_timestamp": str(datetime.now()),
                "sync_method": "manual_cpu_conversion",
            }
        }
        
        import json
        metadata_path = os.path.join(output_dir, "rescue_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(rescue_metadata, f, indent=2)
        logger.info("🚨 RESCUE: Rescue metadata saved")
        
    # Step 4: Barrier and rendezvous (same as original)
    dist.barrier()

    logger.info(f"🎉 RESCUE: Saved model checkpoint to {output_dir} successfully")
    xm.rendezvous("_rescue_save")

ScaleRAETrainer._rescue_save_sharded = _rescue_save_sharded

def sync_to_cpu_alternative(state_dict):
    """
    Alternative sync_to_cpu implementations to work around potential API bugs
    """
    logger.info("Trying alternative sync_to_cpu methods...")
    
    # Method 1: Manual tensor-by-tensor conversion with error handling
    def method1_manual_convert(state_dict):
        logger.info("Method 1: Manual tensor conversion with error handling")
        converted = {}
        failed_keys = []
        
        for key, value in state_dict.items():
            try:
                if isinstance(value, torch.Tensor):
                    # Force XLA mark step before conversion
                    xm.mark_step()
                    # Try direct CPU conversion
                    converted[key] = value.detach().cpu().to(torch.float32)
                    logger.debug(f"  ✅ {key}: {value.shape}")
                else:
                    converted[key] = value
            except Exception as e:
                logger.warning(f"  ❌ Failed to convert {key}: {e}")
                failed_keys.append(key)
                
        if failed_keys:
            logger.warning(f"Failed to convert {len(failed_keys)} tensors: {failed_keys[:5]}...")
        return converted, failed_keys
    
    # Method 2: Chunked conversion (convert small batches)
    def method2_chunked_convert(state_dict):
        logger.info("Method 2: Chunked conversion")
        converted = {}
        chunk_size = 10  # Convert 10 tensors at a time
        
        tensor_items = [(k, v) for k, v in state_dict.items() if isinstance(v, torch.Tensor)]
        non_tensor_items = [(k, v) for k, v in state_dict.items() if not isinstance(v, torch.Tensor)]
        
        # Add non-tensor items first
        for key, value in non_tensor_items:
            converted[key] = value
            
        # Convert tensors in chunks
        for i in range(0, len(tensor_items), chunk_size):
            chunk = tensor_items[i:i+chunk_size]
            logger.info(f"  Converting chunk {i//chunk_size + 1}/{(len(tensor_items)-1)//chunk_size + 1}")
            
            xm.mark_step()  # Ensure XLA operations are complete
            
            for key, tensor in chunk:
                try:
                    converted[key] = tensor.detach().cpu().to(torch.float32)
                except Exception as e:
                    logger.error(f"    Failed to convert {key}: {e}")
                    
        return converted, []
    
    # Method 3: Use XLA's built-in utilities with different flags
    def method3_xla_utilities(state_dict):
        logger.info("Method 3: XLA utilities with different flags")
        
        # Force a compilation barrier
        xm.mark_step()
        xm.wait_device_ops()
        
        # Try with different XLA conversion methods
        converted = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                try:
                    # Method 3a: Use xm.to_device explicitly first
                    device_tensor = xm.to_device(value, torch.device('cpu'))
                    converted[key] = device_tensor.to(torch.float32)
                except Exception as e1:
                    try:
                        # Method 3b: Force sync then convert
                        xm.mark_step()
                        converted[key] = value.detach().cpu().to(torch.float32)
                    except Exception as e2:
                        logger.error(f"Both XLA methods failed for {key}: {e1}, {e2}")
                        raise e2
            else:
                converted[key] = value
                
        return converted, []
    
    # Try methods in order
    methods = [
        ("Manual conversion", method1_manual_convert),
        ("Chunked conversion", method2_chunked_convert), 
        ("XLA utilities", method3_xla_utilities),
    ]
    
    for method_name, method_func in methods:
        try:
            logger.info(f"🔄 Attempting {method_name}...")
            converted_dict, failed_keys = method_func(state_dict)
            
            if not failed_keys:
                logger.info(f"✅ {method_name} succeeded!")
                return converted_dict
            else:
                logger.warning(f"⚠️ {method_name} partially failed")
                
        except Exception as e:
            logger.error(f"❌ {method_name} failed: {e}")
            continue
    
    # If all methods fail, raise the last exception
    raise RuntimeError("All sync_to_cpu alternatives failed!")

def _save_with_alternative_sync(self, output_dir, state_dict=None):
    """
    Alternative _save method using different sync approaches
    """
    logger.info("Calling alternative _save with different sync methods...")
    output_dir = output_dir.replace("gs://", "/mnt/")
    logger.info(f"Saving model checkpoint to {output_dir}")

    state_dict = {"model": self.model.state_dict()}
    
    # Try alternative sync methods instead of the original sync_to_cpu
    try:
        state_dict = sync_to_cpu_alternative(state_dict)
        logger.info("Alternative CPU sync completed successfully")
    except Exception as e:
        logger.error(f"Alternative sync failed, falling back to rescue save: {e}")
        # Fallback to rescue save if sync fails
        self._rescue_save_sharded()
        return

    if dist.get_rank() == 0:
        logger.info("Saving model state dict to CPU")
        if output_dir.startswith("/mnt/") and not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with BytesIO() as buffer, self.fs.open(os.path.join(output_dir, "model.pth.zstd"), 'wb') as f:
            torch.save(state_dict, buffer)
            f.write(zstd.compress(buffer.getvalue()))
        logger.info("Saving model state dict to CPU successfully")
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir.replace("gs://", "/mnt/"))
        logger.info("Saving tokenizer to CPU successfully")
        with self.fs.open(os.path.join(output_dir, "training_args.bin"), 'wb') as f:
            torch.save(self.args, f)
        logger.info("Saving training args to CPU successfully")
        self.model.config.save_pretrained(output_dir.replace("gs://", "/mnt/"))
        logger.info("Saving model config to CPU successfully")
    dist.barrier()

    logger.info(f"Saved model checkpoint to {output_dir} successfully")
    xm.rendezvous("_save_alternative")

# Alternative save method available as:
# ScaleRAETrainer._save = _save_with_alternative_sync

def is_world_process_zero(self):
    return dist.get_rank() == 0

ScaleRAETrainer.is_world_process_zero = is_world_process_zero

def train(self, resume_from_checkpoint=None, *args, **kwargs):
    logger.info("Calling spmd monkey patch for ScaleRAETrainer.train...")
    
    # Log rescue mode status
    if getattr(self.args, 'rescue_ckpt', False):
        logger.info("🚨 RESCUE MODE ACTIVATED: Will perform rescue save after checkpoint loading")
    
    # Normal training flow (rescue logic happens in _load_from_checkpoint)
    if isinstance(resume_from_checkpoint, str) and resume_from_checkpoint.lower() == "true":
        logger.info("resume_from_checkpoint is set to True, resuming from the latest checkpoint")
        tracked_steps = self.checkpoint_manager.all_steps()
        logger.info(f"tracked_steps: {tracked_steps}")
        if tracked_steps:
            max_step = max(tracked_steps)
            resume_from_checkpoint = self.checkpoint_manager._get_path(max_step).replace("gs://", "/mnt/")
            logger.info(f"Max step detected, resuming from checkpoint {resume_from_checkpoint}")
        else:
            logger.warning("No checkpoint found, starting from scratch")
            resume_from_checkpoint = None
    logger.info(f"Start training with resume_from_checkpoint set to: {resume_from_checkpoint}")
    
    super(ScaleRAETrainer, self).train(resume_from_checkpoint=resume_from_checkpoint, *args, **kwargs)




ScaleRAETrainer.train = train
################################################################################

################################################################################
# monkey-patched to spmd_trainer
from scale_rae.train import spmd_trainer
from scale_rae.train.spmd_trainer import get_mm_adapter_state_maybe_zero_3
def safe_save_model_for_hf_trainer(trainer, output_dir):
    logger.info("Calling spmd monkey patch for safe_save_model_for_hf_trainer...")

    output_dir = trainer._get_output_dir(None).rstrip('/') + "-last"
    if dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        output_dir = output_dir.replace("gs://", "/mnt/")

        keys_to_match = ['mm_projector', 'pos_emb', 'vision_sampler', 'vision_sampler_layers', 'vision_query', 'image_newline']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        if dist.get_rank() == 0:
            trainer.model.config.save_pretrained(output_dir.replace("gs://", "/mnt/"))

        if not IS_XLA_AVAILABLE:
            raise NotImplementedError("Only XLA is supported for now.")

        ckpt = {"model": weight_to_save}

        logger.info(f"Saving mm_mlp_adapter to {output_dir}")
        if dist.get_rank() == 0:
            with trainer.fs.open(os.path.join(output_dir, "mm_projector.pth"), 'wb') as f:
                torch.save(ckpt, f)
        dist.barrier()
        xm.rendezvous("save_mm_mlp_adapter")
        return


    if False and getattr(trainer.args, "tune_adapter_and_vision_head", False):
        output_dir = output_dir.replace("gs://", "/mnt/")

        keys_to_match = ['mm_projector', 'vision_head', 'diff_head', "latent_queries"]
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        if dist.get_rank() == 0:
            trainer.model.config.save_pretrained(output_dir.replace("gs://", "/mnt/"))

        if not IS_XLA_AVAILABLE:
            raise NotImplementedError("Only XLA is supported for now.")

        ckpt = {"model": weight_to_save}

        logger.info(f"Saving mm_mlp_adapter to {output_dir}")
        if dist.get_rank() == 0:
            with trainer.fs.open(os.path.join(output_dir, "mmprojector_and_visionhead.pth"), 'wb') as f:
                torch.save(ckpt, f)
        dist.barrier()
        xm.rendezvous("save_mm_mlp_adapter")
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    trainer._save(output_dir)

spmd_trainer.safe_save_model_for_hf_trainer = safe_save_model_for_hf_trainer
################################################################################

################################################################################
# monkey-patched to pytorch 2.5.1 distributed checkpoint for file handling
from torch.distributed.checkpoint.filesystem import _FileSystemWriter, _TensorLoader, _OverlappingCpuLoader, _SerialCpuLoader, _item_size, _write_item, _metadata_fn
from torch.distributed.checkpoint import filesystem
from torch.distributed.checkpoint.planner import SavePlanner, WriteItemType
from typing import Callable, List, cast
import queue
from torch.distributed.checkpoint.storage import WriteResult
from pathlib import Path
from torch.distributed.checkpoint.metadata import Metadata
import pickle

from io import UnsupportedOperation

def _write_files_from_queue(
    create_stream: Callable,
    file_queue: queue.Queue,
    result_queue: queue.Queue,
    planner: SavePlanner,
    inflight_threshhold: int,
    use_fsync: bool,
    thread_count: int,
) -> None:
    try:
        while True:
            file_name, storage_key, write_items = file_queue.get_nowait()
            loader: _TensorLoader

            custom_backend_name = torch._C._get_privateuse1_backend_name()
            custom_device_mod = getattr(torch, custom_backend_name, None)

            # TODO: Using the OverlappingCpuLoader with multiple threads creates significant
            # performance degredation, observed as being related to cuda stream syncs. We
            # should try to fix this and use _OverlappingCpuLoader for all threaded cases
            if (
                thread_count == 1
                and (
                    torch.cuda.is_available()
                    or (custom_device_mod and custom_device_mod.is_available())
                )
                and inflight_threshhold > 0
            ):
                loader = _OverlappingCpuLoader(
                    planner.resolve_data,
                    inflight_threshhold=inflight_threshhold,
                )
            else:
                loader = _SerialCpuLoader(
                    planner.resolve_data,
                )

            tensor_w = [wi for wi in write_items if wi.type != WriteItemType.BYTE_IO]
            for write_item in tensor_w:
                loader.add(_item_size(write_item), write_item)
            loader.start_loading()

            bytes_w = [wi for wi in write_items if wi.type == WriteItemType.BYTE_IO]
            write_results = []

            with create_stream(file_name, "wb") as stream:
                for write_item in bytes_w:
                    data = planner.resolve_data(write_item)
                    write_results.append(
                        _write_item(stream, data, write_item, storage_key)
                    )

                for tensor, write_item in loader.values():
                    assert tensor.is_cpu
                    write_results.append(
                        _write_item(stream, tensor, write_item, storage_key)
                    )

                if use_fsync:
                    try:
                        os.fsync(stream.fileno())
                    except (AttributeError, UnsupportedOperation):
                        os.sync()
            result_queue.put(write_results)
    except queue.Empty:
        pass

filesystem._write_files_from_queue = _write_files_from_queue

def finish(self, metadata: Metadata, results: List[List[WriteResult]]) -> None:
    storage_md = {}
    for wr_list in results:
        storage_md.update({wr.index: wr.storage_data for wr in wr_list})
    metadata.storage_data = storage_md

    metadata.storage_meta = self.storage_meta()

    tmp_path = cast(Path, self.fs.concat_path(self.path, f"{_metadata_fn}.tmp"))
    with self.fs.create_stream(tmp_path, "wb") as metadata_file:
        pickle.dump(metadata, metadata_file)
        if self.sync_files:
            try:
                os.fsync(metadata_file.fileno())
            except (AttributeError, UnsupportedOperation):
                os.sync()

    # delete in-case other checkpoints were present.
    if self.fs.exists(self.metadata_path):
        self.fs.rm_file(self.metadata_path)

    self.fs.rename(tmp_path, self.metadata_path)

_FileSystemWriter.finish = finish
################################################################################

if __name__ == '__main__':

    print("let's launch global training!")
    
    os.environ["SCALE_RAE_LAUNCHER"] = "TORCHXLA_SPMD"
    xr.use_spmd()

    # https://github.com/pytorch/xla/blob/master/docs/source/perf/spmd_distributed_checkpoint.md#process-groups
    # The `xla://` init_method will automatically discover master worker IP, rank,
    # and global world size without requiring environment configuration on TPUs.
    dist.init_process_group(backend="gloo", init_method="xla://")

    assert _LOCAL_PROCESS_GROUP is None
    _LOCAL_PROCESS_GROUP = dist.new_group(backend="gloo", timeout=timedelta(seconds=60))
    logger.info("Init process group done.")

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (num_devices, 1)
    device_ids = np.arange(num_devices)
    mesh = xs.Mesh(device_ids, mesh_shape, ("fsdp", "tp"))
    xs.set_global_mesh(mesh)

    from scale_rae.train.spmd_trainer import train
    train(dist.get_rank())
