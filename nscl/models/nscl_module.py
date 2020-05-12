import torch.nn as nn

from nscl.models.reasoning.reasoning_module import ReasoningModule
from nscl.models.visual.visual_module import VisualModule


class NSCLModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_module = VisualModule()
        self.reasoning_module = ReasoningModule()

    def forward(self, image, question, scene):
        visual_features = self.visual_module(image, scene)
        answer = self.reasoning_module(visual_features, None, question)
        return super().forward(*input)
