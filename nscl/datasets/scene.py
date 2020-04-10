import numpy as np
import json

class Scene(object):
    def __init__(self, json):
        super().__init__()
        self.img_index = json['image_index']
        self.img_filename = json['image_filename']
        self.split = json['split']
        self.objects = Scene.get_objects(json['objects'])
        self.relationships = Relationships(json['relationships'])

    @staticmethod
    def get_objects(objects_json):
        return [Object(json) for json in objects_json]

class Object(object):
    def __init__(self, json):
        super().__init__()
        self.color = json['color']
        self.size = json['size']
        self.shape = json['shape']
        self.material = json['material']

class Relationships(object):
    def __init__(self, json):
        super().__init__()
        self.right = json['right']
        self.behind = json['behind']
        self.front = json['front']
        self.left = json['left']