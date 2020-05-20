class ObjectAnnotation(object):

    def __init__(self, definitions, object_features, attribute_embedding):
        super().__init__()
        all_concepts = [c for concepts in definitions.values() for c in concepts]
        all_attributes = definitions.keys()

        self.num_objects = object_features.shape[0]

        # Compute similarities to all concepts
        self.similarities = dict()
        for c in all_concepts:
            self.similarities[c] = attribute_embedding.similarity(object_features, c)

        # Get all attributes
        self.attributes = []
        for i in range(object_features.shape[0]):
            attribute_map = dict()
            for attr in all_attributes:
                attribute_map[attr] = attribute_embedding.get_attribute(object_features[i], attr)
            self.attributes.append(attribute_map)

    def similarity(self, concept):
        return self.similarities[concept]

    def get_attribute(self, object_id, attr):
        return self.attributes[object_id][attr]