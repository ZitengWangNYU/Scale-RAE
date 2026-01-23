import os
import torch
import torch.nn as nn

from torch.utils.data import Sampler

import dataclasses
import json
from typing import Dict, List, Optional, Union
import numpy as np
import gcsfs
# from google.cloud import storage
import io
import torch_xla.core.xla_model as xm
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
    is_torch_tpu_available
)

from ezcolorlog import root_logger as logger
from scale_rae.utils import IS_XLA_AVAILABLE

from packaging import version
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
from typing import List, Optional

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.utils import is_apex_available
if is_apex_available():
    from apex import amp

import random
from tabulate import tabulate

if IS_XLA_AVAILABLE:
    from torch_xla.experimental.distributed_checkpoint import prime_optimizer

fs = gcsfs.GCSFileSystem(project=os.getenv('GCP_PROJECT', 'your-gcp-project'))

HOME_DIR = os.path.expanduser("~") + "/"
print("HOME_DIR = ", HOME_DIR)

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


def _fetch_gradients(optimizer, param_to_name, selected_module_names):
    gradients = []
    for param_group in optimizer.param_groups:
        for group, params in param_group.items():
            if group == 'params':
                for p in params:
                    # Use the mapping to get the module name
                    module_name = param_to_name.get(p, "")
                    # Check if the module name matches your criteria
                    if isinstance(p, torch.Tensor) and p.grad is not None and any(selected_name in module_name for selected_name in selected_module_names):
                        p.grad = p.grad.to(torch.float32)
                        gradients.append(p.grad.data)
    return gradients

# from torch_xla.core.xla_model import xrt_world_size, all_reduce

REDUCE_SUM = 'sum'
# def reduce_gradients(optimizer, param_to_name, selected_module_names, groups=None, pin_layout=True):
#     count = xrt_world_size()
#     if count > 1:
#         gradients = _fetch_gradients(optimizer, param_to_name, selected_module_names)
#         all_reduce(
#             REDUCE_SUM,
#             gradients,
#             scale=1.0 / count,
#             groups=groups,
#             pin_layout=pin_layout)

def map_params_to_module_names(model_list):
    param_to_name = {}
    for model in model_list:
        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                param_to_name[param] = f"{module_name}.{param_name}"
    return param_to_name


class ScaleRAETrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                # world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps if not os.getenv("SCALE_RAE_LAUNCHER", "") == "TORCHXLA_SPMD" else torch.distributed.get_world_size(),
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
            # loss.backward()

        selected_module_names = ['vision_tower']
        # if self.args.unfreeze_mm_vision_tower:
        #     reduce_gradients(self.optimizer, self.param_to_name, selected_module_names)
        return loss.detach() / self.args.gradient_accumulation_steps

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        opt_model = self.model
        # if self.args.unfreeze_mm_vision_tower:
        #     opt_model.get_model().vision_tower_aux_list = nn.ModuleList(opt_model.get_vision_tower_aux_list())
        #     self.param_to_name = map_params_to_module_names([opt_model])
        verbose = [["name", "dtype", "shape", "trainable?", "lr", "wd"]]
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            assert not (self.args.mm_projector_lr and self.args.mm_vision_sampler_lr)
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]

                for name, param in opt_model.named_parameters():
                    if not param.requires_grad:
                        verbose.append([name, param.shape, param.requires_grad, "N/A.", "N/A."])
                    elif "mm_projector" in name and name in decay_parameters:
                        verbose.append([name, param.shape, param.requires_grad, self.args.mm_projector_lr, self.args.weight_decay])
                    elif "mm_projector" in name and name not in decay_parameters:
                        verbose.append([name, param.shape, param.requires_grad, self.args.mm_projector_lr, "N/A."])
                    elif "mm_projector" not in name and name in decay_parameters:
                        verbose.append([name, param.shape, param.requires_grad, self.args.learning_rate, self.args.weight_decay])
                    elif "mm_projector" not in name and name not in decay_parameters:
                        verbose.append([name, param.shape, param.requires_grad, self.args.learning_rate, "N/A."])
                    else:
                        raise ValueError(f"Unexpected parameter: {name}")

                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]

            elif self.args.diff_head_lr is not None:
                diff_head_parameters = [
                    name for name, _ in opt_model.named_parameters()
                    if "diff_head" in name
                ]

                # ADDED: Check if vision loss is DDT to adjust beta values for diffusion head
                is_ddt_loss = hasattr(self.model, 'vision_loss') and (self.model.vision_loss == 'ddt-loss' or self.model.vision_loss == 'diffusion-loss')
                if is_ddt_loss:
                    print("DDT loss detected: Setting beta2=0.95 for diffusion_head parameters")
    

                for name, param in opt_model.named_parameters():
                    if not param.requires_grad:
                        verbose.append([name, param.shape, False, "N/A.", "N/A."])
                    elif name in diff_head_parameters and name in decay_parameters:
                        verbose.append([name, param.shape, True, self.args.diff_head_lr, self.args.weight_decay])
                    elif name in diff_head_parameters and name not in decay_parameters:
                        verbose.append([name, param.shape, True, self.args.diff_head_lr, "N/A."])
                    elif name not in diff_head_parameters and name in decay_parameters:
                        verbose.append([name, param.shape, True, self.args.learning_rate, self.args.weight_decay])
                    elif name not in diff_head_parameters and name not in decay_parameters:
                        verbose.append([name, param.shape, True, self.args.learning_rate, "N/A."])
                    else:
                        raise ValueError(f"Unexpected parameter: {name}")

                # ADDED: Create base parameter groups for non-diffusion parameters  
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n not in diff_head_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n not in diff_head_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
                
                # ADDED: Create diffusion head parameter groups with conditional beta values
                diff_head_decay_group = {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and n in diff_head_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.diff_head_lr,
                }
                
                diff_head_no_decay_group = {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and n in diff_head_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.diff_head_lr,
                }
                
                # NEW: remember where these two groups will sit in the optimizer so the scheduler can find them later
                first_idx = len(optimizer_grouped_parameters)
                self._diff_head_pg_idx = [first_idx, first_idx + 1]

                # ADDED: Apply custom beta values for DDT loss
                if is_ddt_loss:
                    diff_head_decay_group["betas"] = (0.9, 0.95)  # Custom beta2=0.95 for DDT
                    diff_head_no_decay_group["betas"] = (0.9, 0.95)  # Custom beta2=0.95 for DDT
                
                # ADDED: Add diffusion head groups to optimizer parameters
                optimizer_grouped_parameters.extend([diff_head_decay_group, diff_head_no_decay_group])




            elif self.args.mm_vision_sampler_lr is not None:
                raise NotImplementedError
                vision_sampler_parameters = [name for name, _ in opt_model.named_parameters() if ("vision_sampler" in name) or ("vision_query" in name) ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in vision_sampler_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in vision_sampler_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in vision_sampler_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_vision_sampler_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in vision_sampler_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_vision_sampler_lr,
                    },
                ]
            elif self.args.unfreeze_mm_vision_tower and self.args.mm_vision_tower_lr is not None:

                for name, param in opt_model.named_parameters():
                    if not param.requires_grad:
                        verbose.append([name, param.shape, param.requires_grad, "N/A.", "N/A."])
                    elif "vision_tower" in name and name in decay_parameters:
                        verbose.append([name, param.shape, param.requires_grad, self.args.mm_vision_tower_lr, self.args.weight_decay])
                    elif "vision_tower" in name and name not in decay_parameters:
                        verbose.append([name, param.shape, param.requires_grad, self.args.mm_vision_tower_lr, "N/A."])
                    elif "vision_tower" not in name and name in decay_parameters:
                        verbose.append([name, param.shape, param.requires_grad, self.args.learning_rate, self.args.weight_decay])
                    elif "vision_tower" not in name and name not in decay_parameters:
                        verbose.append([name, param.shape, param.requires_grad, self.args.learning_rate, "N/A."])
                    else:
                        raise ValueError(f"Unexpected parameter: {name}")


                vision_tower_parameters = [name for name, _ in opt_model.named_parameters() if "vision_tower" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in vision_tower_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in vision_tower_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in vision_tower_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_vision_tower_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in vision_tower_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_vision_tower_lr,
                    },
                ]
            else:

                for name, param in opt_model.named_parameters():
                    if not param.requires_grad:
                        verbose.append([name, param.shape, param.requires_grad, "N/A.", "N/A."])
                    elif name in decay_parameters:
                        verbose.append([name, param.shape, param.requires_grad, self.args.learning_rate, self.args.weight_decay])
                    elif name not in decay_parameters:
                        verbose.append([name, param.shape, param.requires_grad, self.args.learning_rate, "N/A."])
                    else:
                        raise ValueError(f"Unexpected parameter: {name}")

                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            # logger.info(f"{tabulate(verbose, headers='firstrow', tablefmt='pipe')}")
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")
        
        
        # if IS_XLA_AVAILABLE:
        #     prime_optimizer(self.optimizer)          # <-- add this line


        # def debug_optimizer_param_mapping():
        #     param_to_name = {p: n for n, p in opt_model.named_parameters()}
        #     param_mapping = {}
        #     state_idx = 0
        #     for group_idx, group in enumerate(self.optimizer.param_groups):
        #         for param_idx, param in enumerate(group['params']):
        #             param_name = param_to_name.get(param, f"unknown_param_{id(param)}")
        #             param_mapping[state_idx] = {
        #                 'name': param_name,
        #                 'group': group_idx,
        #                 'requires_grad': param.requires_grad,
        #                 'shape': param.shape
        #             }
        #             state_idx += 1
            
        #     # Print mapping for debugging
        #     print("Optimizer parameter mapping:")
        #     for idx, info in param_mapping.items():
        #         print(f"  optimizer.state.{idx}: {info['name']} (grad={info['requires_grad']}, shape={info['shape']})")
            
        #     return param_mapping

        # self.optimizer_param_mapping = debug_optimizer_param_mapping()


        return self.optimizer
    

    def remove_prefix(text, prefix='gs://your-bucket/'):
        prefix = f"gs://{os.getenv('GCS_BUCKET_NAME', 'your-bucket')}/"
        if prefix in text:
            return text.replace(prefix, '')
        return text
    
    def _load_rng_state(self, resume_from_checkpoint):
        if resume_from_checkpoint is None:
            return

        # remove local path prefic if exists
        if HOME_DIR in resume_from_checkpoint:
            resume_from_checkpoint_clean = resume_from_checkpoint.replace(HOME_DIR, '')
        else:
            resume_from_checkpoint_clean = resume_from_checkpoint

        resume_from_checkpoint_clean = "gs://" + os.getenv("GCS_BUCKET_NAME", "your-bucket") + "/" + resume_from_checkpoint_clean
        # get worker details
        rank = xm.get_ordinal()
        world_size = xm.xrt_world_size()

        # get path
        RNG_NAME = f'rng_rank-{rank:08d}-of-{world_size:08d}-rng.pth'
        RNG_PATH = os.path.join(resume_from_checkpoint_clean, RNG_NAME)

        # Loading the model weights:
        # client = storage.Client()
        # bucket = client.get_bucket(os.getenv('GCS_BUCKET_NAME', 'your-bucket'))
        # blob = bucket.blob(RNG_PATH)
        # blob_bytes = blob.download_as_bytes()
        # buffer = io.BytesIO(blob_bytes)
        # rng_dict = torch.load(buffer)
        with fs.open(RNG_PATH, 'rb') as f:
            rng_dict = torch.load(f)

        # Setting the seeds correctly
        random.setstate(rng_dict["python"])
        np.random.set_state(rng_dict["numpy"])
        torch.random.set_rng_state(rng_dict["cpu"])
        xm.set_rng_state(rng_dict["xla"])
        print("rng state loaded")

    def _load_optimizer_and_scheduler(self, resume_from_checkpoint):
        if resume_from_checkpoint is None:
            return

        # remove local path prefix
        if HOME_DIR in resume_from_checkpoint:
            resume_from_checkpoint_clean = resume_from_checkpoint.replace(HOME_DIR, '')
        else:
            resume_from_checkpoint_clean = resume_from_checkpoint

        print("resume_from_checkpoint_clean = ", resume_from_checkpoint_clean)

        resume_from_checkpoint_clean = "gs://" + os.getenv("GCS_BUCKET_NAME", "your-bucket") + "/" + resume_from_checkpoint_clean
        
        print("Afterwards resume_from_checkpoint_clean = ", resume_from_checkpoint_clean)

        # get worker details
        rank = xm.get_ordinal()
        world_size = xm.xrt_world_size()

        # get path to file
        WEIGHTS_NAME = "pytorch_model.bin"
        SCHEDULER_NAME = "scheduler.pt"
        SHARD_NAME_OPT = f'opt_rank-{rank:08d}-of-{world_size:08d}-{WEIGHTS_NAME}'
        SHARD_NAME_PATH = os.path.join(resume_from_checkpoint_clean, SHARD_NAME_OPT)
        LR_PATH = os.path.join(resume_from_checkpoint_clean, SCHEDULER_NAME)

        # connect to gcloud bucket
        # client = storage.Client()
        # bucket = client.get_bucket(os.getenv('GCS_BUCKET_NAME', 'your-bucket'))

        # Loading opt state to each device
        # blob = bucket.blob(SHARD_NAME_PATH)
        # blob_bytes = blob.download_as_bytes()
        # buffer = io.BytesIO(blob_bytes)
        # optimizer_state = torch.load(buffer, map_location="cpu")
        with fs.open(SHARD_NAME_PATH, 'rb') as f:
            optimizer_state = torch.load(f, map_location="cpu")
        optimizer_state = optimizer_state['optimizer_state']

        # Loading the schedule to each device
        # blob_lr = bucket.blob(LR_PATH)
        # blob_bytes_lr = blob_lr.download_as_bytes()
        # buffer_lr = io.BytesIO(blob_bytes_lr)
        # lr_scheduler_state = torch.load(buffer_lr)fi
        with fs.open(LR_PATH, 'rb') as f:
            lr_scheduler_state = torch.load(f)

        # No need for this, since already inside XLA spawn?
        # xm.send_cpu_data_to_device(optimizer_state, self.args.device)
        # xm.send_cpu_data_to_device(lr_scheduler_state, self.args.device)

        # Load state
        self.optimizer.load_state_dict(optimizer_state)
        self.lr_scheduler.load_state_dict(lr_scheduler_state)

        logger.info(f"Optimizer state and scheduler successfully loaded from {SHARD_NAME_PATH}")
        print("Loaded optimizer state successfully")

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):

        if resume_from_checkpoint is None:
            return

        # Remove local path (we stored Train State here)
        if HOME_DIR in resume_from_checkpoint:
            resume_from_checkpoint_clean = resume_from_checkpoint.replace(HOME_DIR, '')
        else:
            resume_from_checkpoint_clean = resume_from_checkpoint

        resume_from_checkpoint_clean = "gs://" + os.getenv("GCS_BUCKET_NAME", "your-bucket") + "/" + resume_from_checkpoint_clean
        # Getting worker details
        rank = xm.get_ordinal()
        world_size = xm.xrt_world_size()

        # Getting path to file on bucket
        WEIGHTS_NAME = "pytorch_model.bin"
        SHARD_NAME = f'weights_rank-{rank:08d}-of-{world_size:08d}-{WEIGHTS_NAME}'
        SHARD_NAME_PATH = os.path.join(resume_from_checkpoint_clean, SHARD_NAME)


        # Loading the model weights:
        # client = storage.Client()
        # bucket = client.get_bucket(os.getenv('GCS_BUCKET_NAME', 'your-bucket'))
        # blob = bucket.blob(SHARD_NAME_PATH)
        # blob_bytes = blob.download_as_bytes()
        # buffer = io.BytesIO(blob_bytes)
        # state_dict = torch.load(buffer)
        print("SHARD_NAME_PATH = ", SHARD_NAME_PATH, flush=True)
        with fs.open(SHARD_NAME_PATH, 'rb') as f:
            state_dict = torch.load(f)
        state_dict = state_dict["model"]

        # self.model = self._wrap_model(self.model, )

        # Saving to each worker  - NO NEED TO MOVE ANYTHING TO XLA
        self.model.load_state_dict(state_dict)

    def _save_checkpoint(self, model, trial, metrics=None):
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

        # Names of files
        TRAINING_ARGS_NAME = "training_args.bin"
        WEIGHTS_NAME = "pytorch_model.bin"
        SCHEDULER_NAME = "scheduler.pt"
        TRAINER_STATE_NAME = "trainer_state.json"

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        logger.info(f"Saving model checkpoint to {output_dir}")

        model = self.model
        import torch_xla.core.xla_model as xm
        rank = xm.get_ordinal()
        world_size = xm.xrt_world_size()

        # Name of files to save
        SHARD_NAME = f'weights_rank-{rank:08d}-of-{world_size:08d}-{WEIGHTS_NAME}'
        SHARD_NAME_OPT = f'opt_rank-{rank:08d}-of-{world_size:08d}-{WEIGHTS_NAME}'
        RNG_NAME = f'rng_rank-{rank:08d}-of-{world_size:08d}-rng.pth'

        # Path of files to save
        SHARD_NAME_PATH = os.path.join(output_dir, SHARD_NAME)
        SHARD_NAME_OPT_PATH = os.path.join(output_dir, SHARD_NAME_OPT)
        LR_PATH = os.path.join(output_dir, SCHEDULER_NAME)
        TRAIN_ARGS_PATH = os.path.join(output_dir, TRAINING_ARGS_NAME)
        TRAINER_STATE_NAME_PATH = os.path.join(output_dir, TRAINER_STATE_NAME)
        RNG_PATH = os.path.join(output_dir, RNG_NAME)
        lr_scheduler_state_dict = self.lr_scheduler.state_dict()

        # Final form of model and opt
        ckpt = {
            'model': self.model.state_dict(),
            'shard_metadata': self.model.get_shard_metadata()
        }
        opt_ckpt = {
            'optimizer_state' : self.optimizer.state_dict(),
            'shard_metadata': self.model.get_shard_metadata()
        }

        # Saving model shards
        with fs.open(SHARD_NAME_PATH, 'wb') as f:
            xm.save(ckpt, f, master_only=False)

        # Saving optimizer shards
        with fs.open(SHARD_NAME_OPT_PATH, 'wb') as f:
            xm.save(opt_ckpt, f, master_only=False)

        # saving lr scheduler and train state json
        if xm.is_master_ordinal(local=False):
            with fs.open(LR_PATH, 'wb') as f:
                xm.save(lr_scheduler_state_dict, f, master_only=True)

            json_string = json.dumps(dataclasses.asdict(self.state), indent=2, sort_keys=True) + "\n"
            with fs.open(TRAINER_STATE_NAME_PATH, 'w') as f:
                f.write(json_string)

        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        rng_states["xla"] = xm.get_rng_state()
        with fs.open(RNG_PATH, 'wb') as f:
            torch.save(rng_states, f)

    def get_train_dataloader(self) -> DataLoader:
        out = super().get_train_dataloader()
        return out._loader

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        import torch_xla.core.xla_model as xm
        ckpt_prefix = os.path.join(output_dir, "model_ckpt")
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        
        # Extract the output directory path
        if output_dir is not None:
            # Strip any GCS path prefix to save locally
            if output_dir.startswith('gs://'):
                # Extract just the path after bucket name and any initial directories
                if 'scale-rae/checkpoints/' in output_dir:
                    # Remove the GCS prefix
                    local_path = output_dir.split('scale-rae/checkpoints/')[-1]
                    output_dir = f"checkpoints/{local_path}"
                else:
                    # Just remove the GCS prefix for other paths
                    output_dir = output_dir.split('://')[-1].split('/', 1)[-1]
                    if not output_dir.startswith('checkpoints/'):
                        output_dir = f"checkpoints/{output_dir}"
            
            # Create local directory
            os.makedirs(output_dir, exist_ok=True)


        # os.makedirs(output_dir, exist_ok=True)
        rank = xm.get_ordinal()
        print(rank)
        world_size = xm.xrt_world_size()
        ckpt_path = f'{ckpt_prefix}_rank-{rank:08d}-of-{world_size:08d}.pth'
        state_dict = self.model.state_dict()
        cpu_state_dict = {
                key: value.cpu()
                for key, value in state_dict.items()
            }
        # if not xm.is_master_ordinal(local=False):
        #     cpu_state_dict = {
        #         key:value for key, value in cpu_state_dict.items() if 'vision_tower' not in key
        #     }
        del state_dict
        ckpt = {
            'model': cpu_state_dict,
            'shard_metadata': self.model.get_shard_metadata()
        }
        # os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

        with fs.open(ckpt_path, 'wb') as f:
            xm.save(ckpt, f, master_only=False)
       
        # xm.save(ckpt, ckpt_path, master_only=False)
        print(f'checkpoint saved to {ckpt_path}\n', end='')
        if xm.is_master_ordinal(local=False):
            # consolidate_sharded_model_checkpoints(
            #     ckpt_prefix=ckpt_prefix, ckpt_suffix="_rank-*-of-*.pth", save_path = os.path.join(output_dir, "model_consolidated.pth"))
            # self.model.save_pretrained(output_dir, state_dict=None, safe_serialization=self.args.save_safetensors)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
            TRAINING_ARGS_NAME = "training_args.bin"
            # Good practice: save your training arguments together with the trained model
            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
            self.model.config.save_pretrained(output_dir)

    """Override to add custom logs"""

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_tpu_available():
                import torch_xla.core.xla_model as xm
                xm.mark_step()

            
            # Debug optimizer state

            # opt_state = self.optimizer.state

            # # If this rank has a local‑id 0, print its contents
            # if 0 in opt_state:
            #     st0 = opt_state[0]
            #     print("id=0 slot keys:", st0.keys())      # ['step', 'exp_avg', 'exp_avg_sq', …]
            # else:
            #     print("No local‑id 0 on this rank")
            
            # # End of debug optimizer state




            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            if self.model.vision_loss == 'regression-loss':
                loss_image_mse_scalar = self._nested_gather(self.model.loss_image_mse).mean().item()
                loss_image_cos_scalar = self._nested_gather(self.model.loss_image_cos).mean().item()
                loss_language_scalar = self._nested_gather(self.model.loss_language).mean().item()



            elif self.model.vision_loss == 'diffusion-loss' or self.model.vision_loss == 'ddt-loss':
                loss_image_diff_scalar = self._nested_gather(self.model.loss_image_diff).mean().item()
                loss_language_scalar = self._nested_gather(self.model.loss_language).mean().item()
                # Optional auxiliary regression metrics when enabled alongside diffusion
                aux_mse_scalar = None
                aux_cos_scalar = None
                if hasattr(self.model, "aux_regression_enabled") and self.model.aux_regression_enabled:
                    if hasattr(self.model, "loss_image_mse") and self.model.loss_image_mse is not None:
                        aux_mse_scalar = self._nested_gather(self.model.loss_image_mse).mean().item()
                    if hasattr(self.model, "loss_image_cos") and self.model.loss_image_cos is not None:
                        aux_cos_scalar = self._nested_gather(self.model.loss_image_cos).mean().item()
            else:
                loss_language_scalar = self._nested_gather(self.model.loss_language).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()


            if self.model.vision_loss == 'regression-loss':
                logs["loss_language"] = round(loss_language_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
                logs["loss_image_mse"] = round(loss_image_mse_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
                logs["loss_image_cos"] = round(loss_image_cos_scalar / (self.state.global_step - self._globalstep_last_logged), 4)

            elif self.model.vision_loss == 'diffusion-loss' or self.model.vision_loss == 'ddt-loss':
                logs["loss_language"] = round(loss_language_scalar, 4)
                logs["loss_image_diff"] = round(loss_image_diff_scalar, 4)
                # Also log aux regression losses if present
                if 'aux_mse_scalar' in locals() and aux_mse_scalar is not None:
                    logs["loss_image_mse"] = round(aux_mse_scalar, 4)
                if 'aux_cos_scalar' in locals() and aux_cos_scalar is not None:
                    logs["loss_image_cos"] = round(aux_cos_scalar, 4)
                
            else:
                # loss_language_scalar = self._nested_gather(self.model.loss_languages).mean().item()
                logs["loss_language"] = round(loss_language_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
        


            # Add custom logs
            if self.args.diff_head_lr is not None:
                # NEW: log the current diffusion-head LR using the recorded indices (fallback to pg[2] for backward compatibility)
                if hasattr(self, "_diff_head_pg_idx"):
                    logs["diff_head_lr"] = self.optimizer.param_groups[self._diff_head_pg_idx[0]]["lr"]
                else:
                    logs["diff_head_lr"] = self.optimizer.param_groups[2]["lr"]

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    # NEW -------------------------------------------------------------------
    # Provide a mixed LR scheduler: constant for diffusion head, global policy
    # (e.g. cosine) for everything else. Activated when
    # --diff_head_constant_schedule is passed.
    # -----------------------------------------------------------------------
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """Override to optionally use different LR schedulers for diffusion head vs main model.

        Signature now mirrors HF Trainer.create_scheduler so that
        Trainer.create_optimizer_and_scheduler() can forward the optimizer
        argument without error.
        """

        # Determine if we need special diffusion head handling
        use_special_diff_head_schedule = (
            getattr(self.args, "diff_head_constant_schedule", False) or 
            getattr(self.args, "diff_head_lr_scheduler_type", "cosine") != "cosine"
        )

        # If no special handling needed, delegate to HF Trainer
        if not use_special_diff_head_schedule:
            return super().create_scheduler(num_training_steps, optimizer)

        # Warm-up steps: replicate HF logic (ratio overrides absolute)
        import math
        from torch.optim.lr_scheduler import LambdaLR

        warmup_steps = (
            math.ceil(num_training_steps * self.args.warmup_ratio)
            if self.args.warmup_ratio > 0
            else self.args.warmup_steps
        )

        # Determine the scheduler type for diffusion head
        diff_head_scheduler_type = "cosine"  # default
        
        # Handle backward compatibility
        if getattr(self.args, "diff_head_constant_schedule", False):
            print("WARNING: diff_head_constant_schedule is deprecated. Please use --diff_head_lr_scheduler_type constant_with_warmup instead.")
            diff_head_scheduler_type = "constant_with_warmup"
        else:
            diff_head_scheduler_type = getattr(self.args, "diff_head_lr_scheduler_type", "cosine")

        # Lambda for constant schedule (warmup then constant)
        def _const(step):
            return step / max(1, warmup_steps) if step < warmup_steps else 1.0

        # Lambda for cosine (warmup then decay to 0)
        def _cos(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        # Lambda for cosine with minimum LR (warmup then decay to fraction of peak)
        def _cosine_with_min_lr(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
            min_ratio = getattr(self.args, "diff_head_min_lr_ratio", 0.1)
            cosine_component = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine_component

        # Select the appropriate lambda function for diffusion head
        if diff_head_scheduler_type == "constant_with_warmup":
            diff_head_lambda = _const
        elif diff_head_scheduler_type == "cosine_with_min_lr":
            diff_head_lambda = _cosine_with_min_lr
        else:  # "cosine" or any other value defaults to standard cosine
            diff_head_lambda = _cos

        # Build list of λ, one per param-group
        lambdas = []
        for idx, _ in enumerate(self.optimizer.param_groups):
            lambdas.append(diff_head_lambda if idx in getattr(self, "_diff_head_pg_idx", []) else _cos)

        # Use provided optimizer if any; default to self.optimizer
        opt = self.optimizer if optimizer is None else optimizer

        self.lr_scheduler = LambdaLR(opt, lambdas, last_epoch=-1)
        return self.lr_scheduler
