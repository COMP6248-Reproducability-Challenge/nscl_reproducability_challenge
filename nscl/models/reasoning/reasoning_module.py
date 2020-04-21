import torch
import torch.nn as nn
from nscl.models.embedding.attribute_embedding_space import AttributeEmbeddingSpace
from nscl.models.embedding.relation_embedding_space import RelationEmbeddingSpace
from nscl.models.executor.program_executor import ProgramExecutor

class ReasoningModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.attribute_space = AttributeEmbeddingSpace()
        self.relation_space = RelationEmbeddingSpace()

    def forward(self, visual_features, relation_features, question):
        inputs = []
        buffers = []
        result = None
        executor = ProgramExecutor(visual_features, relation_features, self.attribute_space, self.relation_space)
        for p in question.program:
            if p.operator == 'scene':
                executor.scene()
            elif p.operator == 'query':
                executor.query(*input, p.attribute)
            elif p.operator == 'filter':
                executor.filter(*input, p.attribute, p.concept)
            #TODO: Implement other operators

        return result
