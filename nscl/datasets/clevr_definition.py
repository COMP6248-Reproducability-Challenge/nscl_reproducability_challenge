class CLEVRDefinition(object):

    attribute_concept_map = {
        'color': ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow'],
        'material': ['rubber', 'metal'],
        'shape': ['cube', 'sphere', 'cylinder'],
        'size': ['small', 'large']
    }

    relation_concept_map = {
        'spatial': ['left', 'right', 'front', 'behind']
    }

    @staticmethod
    def get_all_attributes():
        return CLEVRDefinition.attribute_concept_map.keys()

    @staticmethod
    def get_all_relations():
        return CLEVRDefinition.relation_concept_map.keys()

    @staticmethod
    def get_all_concepts():
        return [c for concepts in CLEVRDefinition.attribute_concept_map.values() for c in concepts]

    @staticmethod
    def get_all_relation_concepts():
        return [c for concepts in CLEVRDefinition.relation_concept_map.values() for c in concepts]

    @staticmethod
    def get_attribute_concept_map(attribute):
        return CLEVRDefinition.attribute_concept_map(attribute)

    @staticmethod
    def get_relation_concept_map(relation):
        return CLEVRDefinition.relation_concept_map(relation)

    
