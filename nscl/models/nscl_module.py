import torch.nn as nn

from nscl.models.reasoning.reasoning_module import ReasoningModule
from nscl.models.visual.visual_module import VisualModule


class NSCLModule(nn.Module):
    def __init__(self, definitions, input_dim=256, embedding_dim=64):
        super().__init__()
        self.visual_module = VisualModule()
        self.reasoning_module = ReasoningModule(definitions, input_dim, embedding_dim)

    def forward(self, image, question, scene):
        batch_size = image.size(0)
        answers = []
        visual_features = self.visual_module(image, scene)
        for idx in range(batch_size):
            answers.append(self.reasoning_module(visual_features[idx], None, question[idx]))
        return answers
