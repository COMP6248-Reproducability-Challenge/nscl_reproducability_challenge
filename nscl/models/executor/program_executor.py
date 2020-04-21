import torch
from typing import NewType

Object = NewType('Object', torch.FloatTensor)               # Feature vector
ObjectSet = NewType('ObjectSet', torch.FloatTensor)         # Probability of each object being selected
ObjectConcept = NewType('ObjectConcept', torch.FloatTensor) # Probability of belong to each concept of attribute 'a'
Bool = NewType('Bool', torch.FloatTensor)                   # Probability of yes and no
Count = NewType('Count', torch.IntTensor)                   # Count(single-item tensor)

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

    def scene(self) -> ObjectSet:
        return torch.ones(self.object_features.size(0), dtype=torch.float, device=self.object_features.device)

    def filter(self, object_set: ObjectSet, concept: str) -> ObjectSet:
        mask = self.attribute_embeddings.similarity(self.object_features, concept)
        output = torch.min(object_set, mask)
        return output

    def relate(self, object: Object, relation_concept: str) -> ObjectSet:
        raise NotImplementedError()

    def relate_attribute_equal(self, object: Object, attribute: str) -> ObjectSet:
        raise NotImplementedError()

    def intersect(self, object_set_1: ObjectSet, object_set_2: ObjectSet) -> ObjectSet:
        raise NotImplementedError()

    def union(self, object_set_1: ObjectSet, object_set_2: ObjectSet) -> ObjectSet:
        raise NotImplementedError()

    def query(self, object: Object, attribute: str) -> ObjectConcept:
        raise NotImplementedError()

    def query_attribute_equal(self, obj_1: Object, obj_2: Object, attribute: str) -> Bool:
        raise NotImplementedError()

    def exist(self, object_set: ObjectSet) -> Bool:
        raise NotImplementedError()

    def count(self, object_set: ObjectSet) -> Count:
        raise NotImplementedError()

    def count_less_than(self, object_set: ObjectSet) -> Bool:
        raise NotImplementedError()

    def count_greater_than(self, object_set: ObjectSet) -> Bool:
        raise NotImplementedError()

    def count_equal(self, object_set: ObjectSet) -> Bool:
        raise NotImplementedError()


