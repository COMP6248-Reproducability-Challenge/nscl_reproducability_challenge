import torch
import torch.nn as nn
from nscl.datasets.clevr_definition import CLEVRDefinition

class AttributeOperator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.map = nn.Linear(input_dim, output_dim)

    def forward(self, feature: torch.Tensor):
        out = self.map(feature)
        return out

class ConceptEmbedding(nn.Module):
    def __init__(self, dim, num_attributes):
        super().__init__()
        self.concept_vector = nn.Parameter(torch.randn(dim))
        self.belong_vector = nn.Parameter(torch.randn(num_attributes))

class AttributeEmbeddingSpace(nn.Module):

    # TODO : Adjust the dimension
    INPUT_DIM = 64
    OUTPUT_DIM = 64

    def __init__(self):
        super().__init__()
        self.attribute_operators = dict()
        self.concept_embeddings = dict()
        all_attributes = CLEVRDefinition.get_all_attributes()
        all_concepts = CLEVRDefinition.get_all_concepts()

        for a in all_attributes:
            self.attribute_operators[a] = AttributeOperator(self.INPUT_DIM, self.OUTPUT_DIM)

        for c in all_concepts:
            self.concept_embeddings[c] = ConceptEmbedding(self.OUTPUT_DIM, len(all_attributes))

    """
        object_features : 2D tensor containing visual features of all objects in the scene
                    [
                        [.., .., .., ..], //obj_1 features
                        [.., .., .., ..], //obj_2 features
                        ...
                    ]
        attribute : attribute of interest(color, material, size ...)
        concept : concept to filter(red, green, large ....)

        return : 1D tensor representing the probability of each object is selected
    """
    def similarity(self, object_features: torch.Tensor, concept: str) -> torch.Tensor:
        probs = torch.ones(self.object_features.size(0), dtype=torch.float)
        #TODO : Implement similarity function based on the paper
        return probs

    """
        return : 1D tensor representing the concept(as int) of each object for the given attribute
    """
    def get_attribute(self, object_features: torch.Tensor, attribute: str) -> torch.Tensor:
        concepts = torch.ones(self.object_features.size(0), dtype=torch.int) # concept has to be converted into index 
        #TODO : Implement query function
        return concepts