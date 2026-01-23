import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from .base_encoder import BaseVisionTower
from scale_rae.utils import IS_XLA_AVAILABLE

from ezcolorlog import root_logger as logger

def parse_vae_model_name(model_name):
    """
    Parse VAE model name to extract base model name, resolution, and interpolation size.
    
    Args:
        model_name (str): Model name like "black-forest-labs/FLUX.1-dev-res256-interp64" 
                         or "model-interp64-res256"
    
    Returns:
        tuple: (base_model_name, image_size, interp_size)
    """
    base_name = model_name
    image_size = None
    interp_size = None
    
    # Extract resolution parameter
    if "-res" in model_name:
        parts = model_name.split("-res")
        base_name = parts[0]
        remaining = parts[1]
        res_str = remaining.split("-")[0]
        try:
            image_size = int(res_str)
        except ValueError:
            pass
    
    # Extract interpolation parameter
    if "-interp" in model_name:
        parts = model_name.split("-interp")
        if "-res" not in model_name:  # Only update base name if not already updated by res parsing
            base_name = parts[0]
        interp_str = parts[1].split("-")[0]  # Handle case where interp comes before res
        try:
            interp_size = int(interp_str)
        except ValueError:
            pass
    
    return base_name, image_size, interp_size


class FluxVAEVisionTower(BaseVisionTower):
    def __init__(self, vision_tower_name, args, delay_load=False):
        super(FluxVAEVisionTower, self).__init__(vision_tower_name, args, delay_load)
        
        # Parse model name and extract parameters
        self.base_model_name, self._image_size, self._interp_size = parse_vae_model_name(vision_tower_name)
        
        # Set VAE-specific properties
        self._num_patches_per_side = 64 if self._interp_size is None else int(self._interp_size**0.5)
        self._num_patches = self._num_patches_per_side ** 2
        self._patch_size = 2
        
        self.vision_tower_name = vision_tower_name
        self.is_loaded = False
        self.vae = None

     
        if IS_XLA_AVAILABLE:
            from diffusers.models.attention_processor import AttnProcessor2_0  # type
            import math
            def monkey_patch_call(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
                if len(args) > 0 or kwargs.get("scale", None) is not None:
                            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
                            deprecate("scale", "1.0.0", deprecation_message)

                residual = hidden_states
                if attn.spatial_norm is not None:
                    hidden_states = attn.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )
                if attention_mask is not None:
                    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                    # scaled_dot_product_attention expects attention_mask shape to be
                    # (batch, heads, source_length, target_length)
                    attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
                if attn.group_norm is not None:
                    hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = attn.to_q(hidden_states)

                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)

                inner_dim = key.shape[-1]
                head_dim = inner_dim // attn.heads

                query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                # the output of sdp = (batch, num_heads, seq_len, head_dim)
                query = query.to(value.dtype)
                key = key.to(value.dtype)
                scale = 1.0 / math.sqrt(query.shape[-1])
                attn_scores = torch.matmul(query * scale, key.transpose(-2, -1))  # [B,H,Lq,Lk]
                if attention_mask is not None:
                    attn_scores = attn_scores + attention_mask
                attn_probs = attn_scores.softmax(dim=-1)                          # [B,H,Lq,Lk]
                hidden_states = torch.matmul(attn_probs, value)                   # [B,H,Lq,Dh]

                hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                hidden_states = hidden_states.to(query.dtype)
                # linear proj
                hidden_states = attn.to_out[0](hidden_states)
                # dropout
                hidden_states = attn.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                if attn.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / attn.rescale_output_factor

                return hidden_states

            AttnProcessor2_0.__call__ = monkey_patch_call
            logger.info("Patched AttnProcessor2_0 for XLA checkpointing")


        if not self.delay_load:
            self.load_model()
    
    @property
    def device(self):
        # Force to XLA device if available to avoid FSDP incompatibility
        if IS_XLA_AVAILABLE:
            import torch_xla.core.xla_model as xm
            return xm.xla_device()
        else:
            return super().device
    
    def load_model(self, device_map=None):
        if self.is_loaded:
            return
        
        logger.info(f"Loading FLUX VAE from: {self.base_model_name}")
        
        # Load FLUX VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.base_model_name, 
            subfolder="vae",
            device_map=None,
            low_cpu_mem_usage=False
        )
        
        # Remove quantization layers (not needed for feature extraction)
        self.vae.quant_conv = nn.Identity()
        self.vae.post_quant_conv = nn.Identity()
        
        # Always freeze VAE weights
        self.vae.requires_grad_(False)
        self.vae.eval()

        # Load image processor
        self.image_processor = VaeImageProcessor()
        self.image_processor.crop_size = {
            'height': self._image_size if self._image_size is not None else self.vae.config.sample_size,
            'width': self._image_size if self._image_size is not None else self.vae.config.sample_size
        }
        
        self.is_loaded = True
        logger.info(f"FLUX VAE loaded successfully. Target tokens: {self._interp_size or 'native'}. Image size: {self._image_size}")


        """Monkey patch the VAE encoder to use XLA-compatible checkpointing."""
        # Bind the patched method to the encoder
        if IS_XLA_AVAILABLE and False: # Disable for now as it seems to cause issues
            from torch_xla.utils.checkpoint import checkpoint as xla_checkpoint
            
            def patched_forward(self, sample):
                sample = self.conv_in(sample)
                
                if self.training and self.gradient_checkpointing:
                    # Use XLA checkpoint instead of torch.utils.checkpoint
                    for down_block in self.down_blocks:
                        sample = xla_checkpoint(down_block, sample)
                    sample = xla_checkpoint(self.mid_block, sample)
                else:
                    for down_block in self.down_blocks:
                        sample = down_block(sample)
                    sample = self.mid_block(sample)
                
                # post-process
                sample = self.conv_norm_out(sample)
                sample = self.conv_act(sample)
                sample = self.conv_out(sample)
                
                return sample
        

            import types
            self.vae.encoder.forward = types.MethodType(patched_forward, self.vae.encoder)
            logger.info("VAE encoder patched for XLA checkpointing")

        # Delete decoder to save memory (we only need encoder)
        del self.vae.decoder
        self.vae.decoder = None
        logger.info("VAE decoder removed to save memory")
            
    def patchify(self, latents, p: int):
        # latents: [B, C, H, W]  ->  out: [B, (H*W)//p**2, C*p*p]
        B, C, H, W = latents.shape
        assert H % p == 0 and W % p == 0, "H and W must be multiples of p"
        h, w = H // p, W // p
        # [B, C, H, W] -> [B, C, h, p, w, p] -> [B, h, w, C, p, p] -> [B, h*w, C*p*p]
        out = (
            latents.reshape(B, C, h, p, w, p)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(B, h * w, C * p * p)
        )
        return out

    def _forward(self, images):
        """
        Encode images to VAE latent space and return as sequence tokens.
        
        Args:
            images: [B, 3, H, W] input images
            
        Returns:
            latents: [B, num_tokens, hidden_size] sequence of VAE latent tokens
        """


        with torch.no_grad():
            # Encode to VAE latent space
            latent_dist = self.vae.encode(images).latent_dist
            latents = latent_dist.sample()
            # Apply FLUX scaling (shift and scale)
            latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        # Convert from [B, C, H, W] to [B, H*W, C] sequence format
        B, C, H, W = latents.shape
        if self._patch_size > 1:
            # Use unfold to extract patches
            # latents = F.unfold(latents, kernel_size=self._patch_size, stride=self._patch_size)
            latents = self.patchify(latents, self._patch_size) 

            # latents is now [B, C*patch_size*patch_size, num_patches]
            # latents = latents.transpose(1, 2)  # [B, num_patches, C*patch_size*patch_size]
        else:
            latents = latents.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        # @Austin: I think we should not interpolate here. Note@austin: yeah, it is just a robust handling of the input.
        # Interpolate to target token length if specified
        # if self._interp_size and H*W != self._interp_size:
        #     target_h = target_w = int(self._interp_size**0.5)
        #     # Reshape back to spatial format for interpolation
        #     latents = latents.reshape(B, H, W, C).permute(0, 3, 1, 2)
        #     latents = F.interpolate(latents, size=(target_h, target_w), mode='bilinear', align_corners=False)
        #     # Convert back to sequence format
        #     latents = latents.permute(0, 2, 3, 1).reshape(B, target_h*target_w, C)

        return latents
    
    def decode(self, latents):
        """
        Decode VAE latents back to images.
        
        Args:
            latents: [B, num_tokens, hidden_size] VAE latents
            
        Returns:
            images: [B, 3, H, W] decoded images
        """
        assert len(latents.shape) == 3, "latents should be a 3D tensor"
        if self._patch_size > 1:
            B, L, Cp2 = latents.shape
            p = self._patch_size
            C = Cp2 // (p * p)
            H = W = int((L * (p * p)) ** 0.5)  # assuming square input
            assert L * (p * p) == H * W, "Incompatible dimensions for folding"
            latents = latents.transpose(1, 2)  # [B, C*p*p, (H/p)*(W/p)]
            latents = F.fold(latents, output_size=(H, W), kernel_size=p, stride=p)  # [B, C, H, W]

        # Reverse FLUX scaling
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        
        # Decode to image
        with torch.no_grad():
            images = self.vae.decode(latents, return_dict=False)[0]
        
        return images
if __name__ == "__main__":
    from PIL import Image
    from torchvision.transforms import ToTensor, ToPILImage
    import numpy as np
    # create namespace for args
    args = {'mm_vision_select_layer': 'last', 'mm_vision_select_feature': 'patch', 'unfreeze_mm_vision_tower': False}
    # convert dict to namespace
    from types import SimpleNamespace
    args = SimpleNamespace(**args)
    tower = FluxVAEVisionTower("black-forest-labs/FLUX.1-dev", args, delay_load=False)
    image = '../../../tf.png'
    im = Image.open(image).convert("RGB").resize((256, 256))
    im = ToTensor()(im).unsqueeze(0)
    im = im * 2.0 - 1.0  # normalize to [-1, 1]
    print("Input image shape:", im.shape, 'mean:', im.mean().item(), 'std:', im.std().item())
    z = tower._forward(im)
    print("z shape:", z.shape)
    decoded_im = tower.decode(z)
    print("Decoded image shape:", decoded_im.shape, 'mean:', decoded_im.mean().item(), 'std:', decoded_im.std().item())
    decoded_im = (decoded_im + 1.0) / 2.0  # denormalize to [0, 1]
    # save im
    pil_recon_im = ToPILImage()(decoded_im.squeeze(0).cpu())
    pil_recon_im.save("../../../recon.png")
    print("Decoded image saved as ../../../recon.png")