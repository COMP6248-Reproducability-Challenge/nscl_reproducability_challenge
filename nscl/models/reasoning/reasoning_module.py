import torch.nn as nn

from nscl.models.embedding.attribute_embedding_space import AttributeEmbeddingSpace
from nscl.models.embedding.relation_embedding_space import RelationEmbeddingSpace
from nscl.models.executor.program_executor import ProgramExecutor


class ReasoningModule(nn.Module):
    def __init__(self, definitions, input_dim, embedding_dim):
        super().__init__()
        self.attribute_space = AttributeEmbeddingSpace(definitions, input_dim, embedding_dim)
        self.relation_space = RelationEmbeddingSpace()

    def forward(self, visual_features, relation_features, question):
        input_buffers = []
        executor = ProgramExecutor(visual_features, relation_features, self.attribute_space, self.relation_space)
        for p in question.program:
            inputs = []
            for input_id in p.input_ids:
                inputs.append(input_buffers[input_id])

            if p.operator == 'scene':
                input_buffers.append(executor.scene())
            elif p.operator == 'query':
                input_buffers.append(executor.query(*inputs, p.attribute))
            elif p.operator == 'filter':
                input_buffers.append(executor.filter(*inputs, p.concept))
            elif p.operator == 'unique':
                input_buffers.append(executor.unique(*inputs))
            elif p.operator == 'query_attribute_equal':
                input_buffers.append(executor.query_attribute_equal(*inputs, p.attribute))
            elif p.operator == 'count':
                input_buffers.append(executor.count(*inputs))
            elif p.operator == 'exist':
                input_buffers.append(executor.exist(*inputs))

            # TODO: Implement remaining operators
            else:
                print(f'Operator not implemented {p.operator}')
                break

        result = input_buffers[-1]
        return result
