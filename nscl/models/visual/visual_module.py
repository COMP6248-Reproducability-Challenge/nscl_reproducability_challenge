import torch
import torch.nn as nn

__all__ = ['VisualModule']

class VisualModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Define all models

    def forward(self, img):
        visual_features, relation_features = None, None
        # Do forward pass
        return visual_features, relation_features