import torch

class ProgramExecutor(object):

    """
        object_features : 2D tensor containing visual features of all objects in the scene
                    [
                        [.., .., .., ..], //obj_1 features
                        [.., .., .., ..], //obj_2 features
                        ...
                    ]
        relation_features : tensor containing relation features of all objects in the scene
        attribute_embeddings : embedding space for all object_level attributes(color, material, size ...)
        relation_embeddings : embedding space for all relation_attributes(left, right, ...)
    """
    def __init__(self, object_features, relation_features, attribute_embeddings, relation_embeddings):
        super().__init__()
        self.object_features = object_features
        self.relation_features = relation_features
        self.attribute_embeddings = attribute_embeddings
        self.relation_embeddings = relation_embeddings

    def scene(self):
        return torch.ones(self.object_features.size(0), dtype=torch.float, device=self.object_features.device)

    def filter(self, object_set, attribute, concept):
        raise NotImplementedError()

    def query(self, object_set, attribute):
        raise NotImplementedError()
