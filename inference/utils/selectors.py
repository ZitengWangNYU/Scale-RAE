"""File containing all selectors."""

import warnings
import os
from urllib.request import urlretrieve
from packaging import version
from PIL import Image
from typing import Union

import numpy as np
import scipy
from sklearn.preprocessing import normalize as sk_norm
import torch as th
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoProcessor, AutoModel

import ImageReward as RM


def _pil_transform(px):
    return transforms.Compose([
        transforms.Resize(
            size=px,
            interpolation=transforms.InterpolationMode.BICUBIC,
            max_size=None,
            antialias=True
        ),
        transforms.CenterCrop(size=(px, px)),
        lambda img: img.convert('RGB'),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])


def _tensor_transform(px):
    return transforms.Compose([
        transforms.Resize(
            size=px,
            interpolation=transforms.InterpolationMode.BICUBIC,
            max_size=None,
            antialias=True
        ),
        transforms.CenterCrop(size=(px, px)),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])


def _tensor_to_pil(tensor: th.tensor):
    # normalize to [0, 1]
    tensor = ((tensor - th.min(tensor)) / (th.max(tensor) - th.min(tensor))) * 255.
    tensor = tensor.permute(0, 2, 3, 1).to(
        "cpu", dtype=th.uint8
    ).numpy()
    return [Image.fromarray(t) for t in tensor]


class MLP(th.nn.Module):
    """MLP definition for improved aesthetic model."""
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = th.nn.Sequential(
            th.nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            th.nn.Dropout(0.2),
            th.nn.Linear(1024, 128),
            #nn.ReLU(),
            th.nn.Dropout(0.2),
            th.nn.Linear(128, 64),
            #nn.ReLU(),
            th.nn.Dropout(0.1),

            th.nn.Linear(64, 16),
            #nn.ReLU(),

            th.nn.Linear(16, 1)
        )
  
    def forward(self, x):
        return self.layers(x)


class PriorSelector:
    """Base class for Prior Selector.

    An optimizer is class with three members: load_model_params, get_metric_fn, 
    and select_fn.

    load_model_params loads the model params for calculating the metric.

    get_metric_fn returns a callable function, metric_fn, that calculates the
    metrics on prior to be maximized.

    select_fn performs the selection loop of the prior based on the metrics
    calculated by metric_fn. It by default selects the prior that maximizes
    the metric.
    
    Note that regardless of the metric, the metric_fn should return a scalar
    metric for each prior, and we always select the prior that **minimizes**
    the metric.
    """

    def __init__(self, args, device, debug=False):
        # we expect the config to provide info about the metric to be maximized,
        # as well as the selection loop.
        self.args = args
        self.device = device
        self.debug = debug

    def load_model(self):
        """Load the model params for calculating the metric."""
        raise NotImplementedError('Subclasses must implement load_model_params().')

    def select_by_metric(
        self,
        priors: th.tensor,
        samples: th.tensor,
        metrics: th.tensor,
        target_size: int = 1,
        maximized: bool = False
    ):
        """Perform selection by metrics calculated."""
        assert metrics.ndim == 1, "Metric should be 1-D tensor"
        assert metrics.shape[0] == samples.shape[0], "Number of metrics should match prior."
        if not maximized:
          # we negate to get minimized metrics
          metrics = metrics * -1
        top_return = th.topk(metrics, target_size)
        top_metrics, top_indices = top_return.values, top_return.indices
        # we negate the metric back to get the ground truth metric.
        if not maximized:
          # we negate it back to get ground truth metric
          top_metrics = top_metrics * -1
        return priors[top_indices.to(priors.device)], samples[top_indices.to(samples.device)], top_metrics


    def select(
        self,
        priors: th.tensor,
        samples: th.tensor,
        prompt: str,
        interm: th.tensor,
        keep_num: int = 1
    ):
        """Returns the selected prior.
        
        Args:
          priors: All the priors to be selected from.
          metrics: the metric corresponding to each prior. We strictly require all
            metric functions to produce a metric to be **minimized**, so here we
            negate the metric to select the smallest.
        """
        raise NotImplementedError('Subclasses must implement select().')


class ImageRewardSelector(PriorSelector):
    """Return prior with best Image Reward Score."""
    def __init__(self, args, device, debug=False, opt=False):
        super().__init__(args, device, debug)
        self.model = self.load_model()
        self.tensor_transform = _tensor_transform(224)
        self.opt = opt
        # RM model mean & std
        self.mean = 0.16717362830052426
        self.std = 1.0333394966054072

    def load_model(self):
        model = RM.load('ImageReward-v1.0', download_root=os.path.expanduser('~/.cache/ImageReward'))  # pylint: disable=c-extension-no-member
        return model
    
    def requires_grad_(self, requires_grad=True):
        self.model.requires_grad_(requires_grad)

    def select(
        self,
        priors: th.tensor,
        samples: th.tensor,
        prompt: str,
        interm: th.tensor,
        keep_num: int = 1
    ):
        text_input = self.model.blip.tokenizer(
            prompt,  # all input prompt should be the same
            padding='max_length',
            truncation=True,
            max_length=35,
            return_tensors="pt"
        ).to(self.device)
        # ensure to reproduce the Image Reward result
        if self.opt:
            images = self.tensor_transform(samples)
        else:
            if isinstance(samples, th.Tensor):
                images = _tensor_to_pil(samples)
            elif isinstance(samples, list):
                images = samples
            else:
                images = [samples]

        txt_set = []
        for sample in images:
            image_embeds = self.model.blip.visual_encoder(
                _pil_transform(224)(sample).unsqueeze(0).to(self.device)
            )
            # text encode cross attention with image
            image_atts = th.ones(image_embeds.size()[:-1],dtype=th.long).to(self.device)
            text_output = self.model.blip.text_encoder(
                text_input.input_ids,
                attention_mask=text_input.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            txt_set.append(text_output.last_hidden_state[:, 0, :])

        txt_features = th.cat(txt_set, 0).float() # [image_num, feature_dim]
        rewards = self.model.mlp(txt_features) # [image_num, 1]
        rewards = (rewards - self.mean) / self.std
        rewards = th.squeeze(rewards, dim=1)
        _, rank = th.sort(rewards, dim=0, descending=True)
        _, indices = th.sort(rank, dim=0)
        if self.debug:
            return rewards

        best_prior, best_sample, best_metrics = self.select_by_metric(
            priors, samples, rewards, target_size=keep_num, maximized=True
        )
        return best_prior, best_sample, best_metrics


if __name__ == "__main__":

    model = ImageRewardSelector(None, device='cuda', debug=True)