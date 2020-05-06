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
        input_buffers = []
        executor = ProgramExecutor(visual_features, relation_features, self.attribute_space, self.relation_space)
        for p in question.program:
            inp = input_buffers[p.input_id] if p.operator != 'scene' else None
            if p.operator == 'scene':
                input_buffers.append(executor.scene())
            elif p.operator == 'query':
                input_buffers.append(executor.query(inp, p.attribute))
            elif p.operator == 'filter':
                input_buffers.append(executor.filter(inp, p.attribute, p.concept))
            elif p.oprator == 'unique':
                input_buffers.append(executor.unique(inp))
            #TODO: Implement other operators

        result = input_buffers[-1]
        return result
