"""Utility modules for Scale-RAE inference."""

from .load_model import load_scale_rae_model

# ImageRewardSelector is imported only when needed (in scaling_experiment.py)
# to avoid requiring ImageReward package for basic CLI usage

__all__ = [
    "load_scale_rae_model",
]

