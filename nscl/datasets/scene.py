import numpy as np
import torch

try:
    from coco.PythonAPI.pycocotools import mask as mask_utils
except:
    from pycocotools import mask as mask_utils

__all__ = ['Scene', 'Object', 'Relationships']


class Scene(object):
    def __init__(self, json):
        super().__init__()
        self.img_index = json['image_index']
        self.img_filename = json['image_filename']
        self.split = json['split']
        self.objects = Scene.get_objects(json['objects'])
        self.relationships = Relationships(json['relationships'])
        self.detection = json['objects_detection'] if 'objects_detection' in json else json['objects']

        # change box positions to be the same scale as transformed CLEVR image
        bbox = [mask_utils.toBbox(d['mask']) for d in self.detection]
        self.boxes = torch.tensor(transform_bbox(np.array(bbox), 0.8), dtype=torch.float32)

        # Rearrange and filter objects based on detected boxes
        self.rearranged_objects = []
        for x, y, width, height in bbox:
            obj = [obj for obj in self.objects if
                   x < obj.coordinates[0] < x + width and y < obj.coordinates[1] < y + height]
            if len(obj) == 1:
                self.rearranged_objects.append(obj[0])

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
        self.coordinates = json['pixel_coords']


class Relationships(object):
    def __init__(self, json):
        super().__init__()
        self.right = json['right']
        self.behind = json['behind']
        self.front = json['front']
        self.left = json['left']


def transform_bbox(_bbox, scale_factor):
    bbox = _bbox.copy()
    bbox[:, 0] *= scale_factor
    bbox[:, 1] *= scale_factor
    bbox[:, 2] = (_bbox[:, 0] + _bbox[:, 2]) * scale_factor
    bbox[:, 3] = (_bbox[:, 1] + _bbox[:, 3]) * scale_factor
    return bbox
