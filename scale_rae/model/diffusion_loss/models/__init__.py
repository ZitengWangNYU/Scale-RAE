from .lightningDiT import LightningDiT, LightningDDT
# from .nextDiT import LuminaNextDiT2DModel



DiT_ARCH = {
    "DiT": LightningDiT,
    "DDT": LightningDDT,
    # "xattnDiT": LuminaNextDiT2DModel,
}