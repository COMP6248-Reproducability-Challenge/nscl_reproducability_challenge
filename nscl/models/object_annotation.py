class ObjectAnnotation(object):

    def __init__(self, definitions, object_features, attribute_embedding):
        super().__init__()
        self.device = object_features.device
        self.all_concepts = [c for concepts in definitions.values() for c in concepts]
        self.all_attributes = definitions.keys()

        self.num_objects = object_features.shape[0]

        # Compute similarities to all concepts
        object_embeddings = attribute_embedding.map_to_embeddings(object_features)
        self.similarities = dict()
        for c in self.all_concepts:
            self.similarities[c] = attribute_embedding.similarity(object_embeddings, c)

        # Get all attributes
        self.attributes = []
        for i in range(object_features.shape[0]):
            attribute_map = dict()
            for attr in self.all_attributes:
                attribute_map[attr] = attribute_embedding.get_attribute(object_embeddings[i], attr)
            self.attributes.append(attribute_map)

    def similarity(self, concept):
        return self.similarities[concept]

    def get_attribute(self, object_id, attr):
        return self.attributes[object_id][attr]