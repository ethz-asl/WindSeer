import torch
import torch.nn as nn
import torch.nn.functional  as F

"""
2D GAN for the generation of extra training samples for data augmentation.
"""

class Generator(nn.Module):
