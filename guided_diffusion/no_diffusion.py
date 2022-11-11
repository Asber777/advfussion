"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py
Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch as th
from .respace import SpacedDiffusion
from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood


class NoDiffusion(SpacedDiffusion):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    