import torch
import torch.nn.functional as F
from open_clip import create_model_from_pretrained 

from scale_rae.utils import IS_XLA_AVAILABLE

from .base_encoder import ProcessorWrapper
from .clip_encoder import ClipVisionTower

from transformers import AutoModel, AutoProcessor, SiglipImageProcessor

import torch.nn as nn


def parse_model_name(input_string):
    """
    Extract model name and interp value from a model string.
    
    Args:
        input_string (str): The model string to parse
        
    Returns:
        tuple: (model_name, interp_value) where interp_value is an int or None
    """
    # Clean up any whitespace
    input_string = input_string.strip()
    
    # Check if the string contains "-interp"
    if "-interp" in input_string:
        # Split by "-interp"
        parts = input_string.split("-interp")
        model_name = parts[0]
        interp_value = int(parts[1]) if parts[1] else None
    else:
        # No interp value found
        model_name = input_string
        interp_value = None
    
    return model_name, interp_value



def extract_res_interp(model_name):


    if '384' in model_name:
        res = 384
    elif "224" in model_name:
        res = 224
    elif "256" in model_name:
        res = 256
    else:
        res = 224

    # res = 384 if '384' in model_name else 224

    base_model_name, interp = parse_model_name(model_name)


    return base_model_name, res, interp


class SiglipVisionTower(ClipVisionTower):
    def __init__(self, vision_tower_name, args, delay_load=False):
        super(ClipVisionTower, self).__init__(vision_tower_name, args, delay_load)
        base_model_name, res, interp = extract_res_interp(vision_tower_name)

        
        self.vision_tower_name = base_model_name
        self._image_size = res if res is not None else 512
        self._interp_size = interp
        self.normalize_vision = getattr(args, 'normalize_vision', True)
        


        if not self.delay_load:
            self.load_model()
        elif self.unfreeze_mm_vision_tower:
            self.load_model()
        else:

            if "base" in self.vision_tower_name:
                self._hidden_size = 768
            else:
                self._hidden_size = 1152
        



    def load_model(self, device_map=None):
        self.vision_model = "siglip"

        vision_tower = AutoModel.from_pretrained(self.vision_tower_name)
        self.vision_tower = vision_tower.vision_model
        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)

        self.image_processor.crop_size = {
            'height': self._image_size,
            'width': self._image_size
        }

        # exit()

        # clip_model, processor = create_model_from_pretrained(self.vision_tower_name)

        # self.vision_tower = clip_model.visual.trunk
        # self.vision_tower.output_tokens = True

        # self._hidden_size 
        self._hidden_size = self.vision_tower.embeddings.patch_embedding.out_channels
        # self._image_size = self.vision_tower.patch_embed.img_size[0]
        # self._patch_size = self.vision_tower.patch_embed.patch_size[0]
        # self.image_processor = ProcessorWrapper(processor, height=self._image_size, width=self._image_size)

        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        self.is_loaded = True





    def interpolate(self, image_features):
        if self._interp_size is None:
            return image_features

        b, num_tokens, dim = image_features.shape

        if num_tokens != self.num_patches:
            raise NotImplementedError('Non')
            target_h = target_w = int(self._interp_size ** 0.5)
            h = w = int(num_tokens ** 0.5)

# 
            image_features = image_features.view(b, h, w, dim)
            image_features = image_features.permute(0, 3, 1, 2).contiguous()

            image_features = F.interpolate(
                image_features.to(torch.float32),
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            ).to(image_features.dtype)

            # Permute the dimensions back to (b, target_h, target_w, dim)
            image_features = image_features.permute(0, 2, 3, 1).contiguous()

            # Flatten the spatial dimensions (target_h, target_w) into a single dimension
            image_features = image_features.flatten(1, 2)


        # We use layernorm instead of F.normalize
        # if self.normalize_vision:
        #     image_features = F.normalize(image_features, p=2, dim=-1)

        if self.normalize_vision:
            # image_features = self.layer_norm(image_features)
            image_features = F.layer_norm(image_features, (self._hidden_size,), weight=None, bias=None, eps=1e-6)


        return image_features

    @property
    def device(self):
        if IS_XLA_AVAILABLE:
            import torch_xla.core.xla_model as xm
            return xm.xla_device()
        else:
            return super().device

    def feature_select(self, image_forward_outs):

        image_features = image_forward_outs.hidden_states[self.select_layer]

        
        
        return image_features

    def _forward(self, images, interpolate_token = 576):
        if IS_XLA_AVAILABLE:
            from torch_xla.utils.checkpoint import checkpoint
            self.vision_tower.encoder._gradient_checkpointing_func = checkpoint

        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):

            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            interp_features = self.interpolate(image_features)
            return interp_features
