import os
import copy

from ezcolorlog import root_logger as logger

from .siglip_encoder import SiglipVisionTower
from .dino_encoder import DinoVisionTower
from .vae_encoder import FluxVAEVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    """Build vision tower from configuration."""
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    if vision_tower is None or not isinstance(vision_tower, str):
        raise ValueError(f'Vision Tower is not specified in the config: {vision_tower_cfg}')
    
    # SigLIP Vision Towers
    if "google" in vision_tower.lower() and "siglip" in vision_tower.lower():
        logger.info(f"Loading **Google SigLIP** Vision Tower: {vision_tower}")
        return SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "siglip" in vision_tower.lower():
        logger.info(f"Loading **SigLIP** Vision Tower: {vision_tower}")
        return SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    # DINO/WebSSL Vision Towers
    if "dinov2" in vision_tower.lower() or "dino" in vision_tower.lower() or "webssl" in vision_tower.lower():
        logger.info(f"Loading **DINO/WebSSL** Vision Tower: {vision_tower}")
        return DinoVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    # VAE Vision Towers for generation alignment
    if any(diffusion_name in vision_tower.lower() for diffusion_name in ["flux", "stable-diffusion"]):
        logger.info(f"Loading **FLUX VAE** Vision Tower: {vision_tower}")
        return FluxVAEVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}. Supported encoders: SigLIP, DINO/WebSSL, FLUX VAE.')


def build_vision_tower_aux_list(vision_tower_cfg, **kwargs):
    """Build auxiliary vision tower list from configuration."""
    vision_tower_aux_name_list = getattr(vision_tower_cfg, 'mm_vision_tower_aux_list', getattr(vision_tower_cfg, 'vision_tower_aux_list', None))
    vision_tower_aux_token_len_list = getattr(vision_tower_cfg, 'mm_vision_tower_aux_token_len_list', getattr(vision_tower_cfg, 'vision_tower_aux_token_len_list', None))
    vision_tower_aux_list = []
    
    for vision_tower_aux_name, vision_tower_aux_token_len in zip(vision_tower_aux_name_list, vision_tower_aux_token_len_list):
        config = copy.deepcopy(vision_tower_cfg)
        vision_tower_aux_name += "-interp{}".format(vision_tower_aux_token_len)
        
        # SigLIP Vision Towers
        if "google" in vision_tower_aux_name.lower() and "siglip" in vision_tower_aux_name.lower():
            logger.info(f"Loading **Google SigLIP** Vision Tower: {vision_tower_aux_name}")
            vision_tower_aux_list.append(SiglipVisionTower(vision_tower_aux_name, args=config, **kwargs))
        elif "siglip" in vision_tower_aux_name.lower():
            logger.info(f"Loading **SigLIP** Vision Tower: {vision_tower_aux_name}")
            vision_tower_aux_list.append(SiglipVisionTower(vision_tower_aux_name, args=config, **kwargs))

        # DINO/WebSSL Vision Towers
        elif "dinov2" in vision_tower_aux_name.lower() or "dino" in vision_tower_aux_name.lower() or "webssl" in vision_tower_aux_name.lower():
            logger.info(f"Loading **DINO/WebSSL** Vision Tower: {vision_tower_aux_name}")
            vision_tower_aux_list.append(DinoVisionTower(vision_tower_aux_name, args=config, **kwargs))

        # VAE Vision Towers for generation alignment
        elif "flux" in vision_tower_aux_name.lower() or "stable-diffusion" in vision_tower_aux_name.lower():
            logger.info(f"Loading **FLUX VAE** Vision Tower: {vision_tower_aux_name}")
            vision_tower_aux_list.append(FluxVAEVisionTower(vision_tower_aux_name, args=config, **kwargs))
        
        else:
            raise ValueError(f'Unknown vision tower: {vision_tower_aux_name}. Supported encoders: SigLIP, DINO/WebSSL, FLUX VAE.')
    
    return vision_tower_aux_list
