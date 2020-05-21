import torch.nn as nn

from nscl.models.reasoning.reasoning_module import ReasoningModule
from nscl.models.visual.visual_module import VisualModule
from nscl.models.object_annotation import ObjectAnnotation
from nscl.models.embedding.attribute_embedding_space import AttributeEmbeddingSpace

class NSCLModule(nn.Module):
    def __init__(self, definitions, input_dim=256, embedding_dim=64):
        super().__init__()
        self.definitions = definitions
        self.visual_module = VisualModule()
        self.attribute_space = AttributeEmbeddingSpace(definitions, input_dim, embedding_dim)
        self.reasoning_module = ReasoningModule()

    def forward(self, image, question, scene):
        batch_size = image.size(0)
        answers = []
        object_annotations = []
        visual_features = self.visual_module(image, scene)
         
        for idx in range(batch_size):
            object_annotation = ObjectAnnotation(self.definitions, visual_features[idx], self.attribute_space)
            answers.append(self.reasoning_module(question[idx], object_annotation))
            object_annotations.append(object_annotation)

        return object_annotations, answers
