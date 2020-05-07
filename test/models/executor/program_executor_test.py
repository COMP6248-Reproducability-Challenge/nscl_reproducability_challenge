import unittest
from unittest.mock import Mock
import torch
from nscl.models.executor.program_executor import ProgramExecutor
from nscl.models.embedding.attribute_embedding_space import AttributeEmbeddingSpace
from nscl.models.embedding.relation_embedding_space import RelationEmbeddingSpace

attribute_concept_map = {'color':['red', 'green', 'blue', 'yellow']}
num_objects, num_features, num_output = 4, 128, 16

class ProgramExecutorTest(unittest.TestCase):

    def test_execute_scene(self):
        object_features = torch.rand(num_objects, num_features)
        relation_features = torch.rand(num_objects, num_objects)
        attribute_embedding = AttributeEmbeddingSpace(attribute_concept_map, num_features, num_output)
        relation_embedding = RelationEmbeddingSpace()
        executor = ProgramExecutor(object_features, relation_features, attribute_embedding, relation_embedding)
        output = executor.scene()
        self.assertTrue(torch.equal(torch.ones(num_objects), output))

    def test_execute_filter(self):
        concept = 'red'
        object_features = torch.rand(num_objects, num_features)
        relation_features = torch.rand(num_objects, num_objects)
        filter_input = torch.tensor([0.02, 1.0, 1.0, 1.0], dtype=torch.float)
        embedding_output = torch.tensor([0.1, 0.75, 0.9, 0.01]) # object 2 and 3 color is read
        attribute_embedding = AttributeEmbeddingSpace(attribute_concept_map, num_features, num_output)
        relation_embedding = RelationEmbeddingSpace()
        attribute_embedding.similarity = Mock(return_value=embedding_output)
        executor = ProgramExecutor(object_features, relation_features, attribute_embedding, relation_embedding)
        output = executor.filter(filter_input, concept)
        self.assertTrue(torch.equal(torch.tensor([0.02, 0.75, 0.9, 0.01]), output))

    # def test_execute_query(self):
    #     num_objects, num_features = 4, 128
    #     attribute = 'material'

    def test_execute_intersect(self):
        object_features = torch.rand(num_objects, num_features)
        relation_features = torch.rand(num_objects, num_objects)
        attribute_embedding = AttributeEmbeddingSpace(attribute_concept_map, num_features, num_output)
        relation_embedding = RelationEmbeddingSpace()
        executor = ProgramExecutor(object_features, relation_features, attribute_embedding, relation_embedding)
        object_set_1 = torch.tensor([0.2, 0.0, 1.0, 0.5], dtype=torch.float)
        object_set_2 = torch.tensor([0.5, 1.0, 0.7, 0.5], dtype=torch.float)
        output = executor.intersect(object_set_1, object_set_2)
        self.assertTrue(torch.equal(torch.tensor([0.2, 0.0, 0.7, 0.5]), output))

    def test_execute_union(self):
        object_features = torch.rand(num_objects, num_features)
        relation_features = torch.rand(num_objects, num_objects)
        attribute_embedding = AttributeEmbeddingSpace(attribute_concept_map, num_features, num_output)
        relation_embedding = RelationEmbeddingSpace()
        executor = ProgramExecutor(object_features, relation_features, attribute_embedding, relation_embedding)
        object_set_1 = torch.tensor([0.2, 0.0, 1.0, 0.5], dtype=torch.float)
        object_set_2 = torch.tensor([0.5, 1.0, 0.7, 0.5], dtype=torch.float)
        output = executor.union(object_set_1, object_set_2)
        self.assertTrue(torch.equal(torch.tensor([0.5, 1.0, 1.0, 0.5]), output))