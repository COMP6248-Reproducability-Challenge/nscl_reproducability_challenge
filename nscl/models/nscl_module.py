import torch
import torch.nn as nn
from nscl.models.visual.visual_module import VisualModule
from nscl.models.reasoning.reasoning_module import ReasoningModule

class NSCLModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_module = VisualModule()
        self.reasoning_module = ReasoningModule()


    def forward(self, image, question, scene):
        visual_features, relation_features = self.visual_module(image)
        answer = self.reasoning_module(visual_features, relation_features, question)
        return super().forward(*input)