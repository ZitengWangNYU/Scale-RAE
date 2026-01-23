from typing import Optional
from .decoder import GeneralDecoder
from transformers import AutoConfig, AutoImageProcessor, SiglipModel
from torch import nn
import torch
import torch.nn.functional as F


class MultimodalDecoder(nn.Module):
    """
    Decoder that reconstructs images from vision encoder features.
    
    Dynamically adapts to different encoder architectures (SigLIP, WebSSL DINO, etc.)
    by reading encoder config to set correct input dimensions.
    """
    def __init__(self, pretrained_encoder_path: str, general_decoder_config: str, num_patches: int, drop_cls_token: bool = False, decoder_path: Optional[str] = None):
        super().__init__()
        config = AutoConfig.from_pretrained(general_decoder_config)
        self.drop_cls_token = drop_cls_token
        
        # Get image normalization from encoder's processor
        image_processor = AutoImageProcessor.from_pretrained(pretrained_encoder_path)
        self.image_mean = torch.tensor(image_processor.image_mean).view(1, 3, 1, 1)
        self.image_std = torch.tensor(image_processor.image_std).view(1, 3, 1, 1)
        
        # Override decoder input size with encoder's output size
        encoder_config = AutoConfig.from_pretrained(pretrained_encoder_path)
        if hasattr(encoder_config, 'vision_config'):
            config.hidden_size = encoder_config.vision_config.hidden_size  # SigLIP: 1152
        else:
            config.hidden_size = encoder_config.hidden_size  # WebSSL DINO: 1024
        
        self.decoder = GeneralDecoder(config, num_patches=num_patches)
        
        if decoder_path is not None:
            state_dict = torch.load(decoder_path, map_location="cpu")
            keys = self.decoder.load_state_dict(state_dict, strict=False)
            print(f"Decoder loaded successfully")
    def forward(self, zs: torch.Tensor) -> torch.Tensor:
        """Decode vision encoder features to reconstructed images."""
        decoder_output = self.decoder(zs, drop_cls_token=self.drop_cls_token)
        logits = decoder_output.logits
        xs_recon = self.decoder.unpatchify(logits)
        xs_recon = xs_recon * self.image_std + self.image_mean
        return xs_recon
    
    @torch.no_grad()
    def infer(self, zs: torch.Tensor) -> torch.Tensor:
        """Decode vision encoder features (inference mode, no gradients)."""
        decoder_output = self.decoder(zs, drop_cls_token=self.drop_cls_token)
        logits = decoder_output.logits
        xs_recon = self.decoder.unpatchify(logits)
        xs_recon = xs_recon * self.image_std + self.image_mean
        return xs_recon
    

class SigLIPEncoderForDebugging(nn.Module):
    def __init__(self, model_name="google/siglip-so400m-patch14-384", num_tokens=256):
        super().__init__()
        self.model_name = model_name
        self.num_tokens = num_tokens
        self.hidden_size = 1152  # SigLIP-SO400M hidden size        
        self.load_model()
        self.vision_tower.eval()
        image_mean = self.processor.image_mean
        image_std = self.processor.image_std
        self.register_buffer("image_mean", torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(image_std).view(1, 3, 1, 1))
    def load_model(self):
        model = SiglipModel.from_pretrained(self.model_name)
        processor = SiglipImageProcessor.from_pretrained(self.model_name)
        
        self.vision_tower = model.vision_model
        self.processor = processor
    @torch.no_grad() # encoder is always frozen
    def forward(self, images):
        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = (images - self.image_mean) / self.image_std
        outputs = self.vision_tower(images, output_hidden_states=True, interpolate_pos_encoding = True)
        image_features = outputs.hidden_states[-1]
        b, num_tokens, dim = image_features.shape
        h = w = int(num_tokens**0.5)
        target_h = target_w = int(self.num_tokens**0.5)

        if self.num_tokens!=729:
            image_features = image_features.view(b, h, w, dim)
            image_features = image_features.permute(0, 3, 1, 2)
            image_features = F.interpolate(image_features, size=(target_h, target_w), mode='bilinear', align_corners=False)
            image_features = image_features.permute(0, 2, 3, 1).contiguous().view(b, self.num_tokens, dim)

        # Normalize vision if needed
        image_features = F.layer_norm(image_features, (self.hidden_size,), eps =1e-6)
        # append a dumb cls token
        empty_cls = torch.zeros((image_features.shape[0], 1, image_features.shape[-1]), device=image_features.device)
        image_features = torch.cat([empty_cls, image_features], dim=1)
        return image_features
    @torch.no_grad()
    def encode_image(self, image):

        features = self(image)
        
        return features
