import json
import os.path as osp

from PIL import Image
from torch.utils.data import Dataset

from nscl.datasets.scene import Scene
from coco.PythonAPI.pycocotools import mask as mask_utils


class CLEVRObjectDataset(Dataset):

    def __init__(self, img_root, scene_json, img_transform=None, max_size=100000):
        super().__init__()
        self.obj_image = []
        self.label = []
        self.count = 0
        img_location = img_root
        raw_scenes = json.load(open(scene_json))['scenes']

        for raw_scene in raw_scenes:
            scene = Scene(raw_scene)
            img = Image.open(osp.join(img_location, scene.img_filename)).convert('RGB')
            boxes = [mask_utils.toBbox(d['mask']) for d in scene.detection]

            for box in boxes:
                x, y, width, height = box[0], box[1], box[2], box[3]
                crop_image = img_transform(img.crop((x, y, x + width, y + height)))
                obj = next((obj for obj in scene.objects if x < obj.coordinates[0] < x + width
                            and y < obj.coordinates[1] < y + height), None)

                if obj is not None and self.count < max_size:
                    self.obj_image.append(crop_image)
                    self.label.append(obj.shape)
                    self.count += 1

            if self.count >= max_size:
                break

        self.classes = list(set(self.label))

    def __getitem__(self, index):
        return self.obj_image[index], self.classes.index(self.label[index])

    def __len__(self):
        return self.count
