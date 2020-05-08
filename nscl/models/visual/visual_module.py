import os
import os.path as osp

import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet34
from torchvision.ops import RoIAlign

from coco.PythonAPI.pycocotools import mask as mask_utils

__all__ = ['VisualModule']

from nscl.datasets.clevr_dataset import build_clevr_dataset


class VisualModule(nn.Module):

    def __init__(self, ):
        super().__init__()
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.pool_size = 7
        self.downsample_rate = 16
        self.roi_align = RoIAlign(self.pool_size, 1.0/self.downsample_rate, 2)

        self.resnet = resnet34(pretrained=True)
        self.resnet_feature_extractor = nn.Sequential(*list(self.resnet.children())[:-3])
        self.resnet_feature_extractor.eval()

    def forward(self, data):
        images, questions, scenes = data
        boxes = [mask_utils.toBbox(d['mask']) for d in scenes.detection]
        a_boxes = torch.tensor([[0, box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in boxes])
        image_feature = self.resnet_feature_extractor(self.preprocess(images).unsqueeze(0))
        box_feature = self.roi_align(image_feature, a_boxes)
        return image_feature, box_feature


train_img_root = '/Users/mark/Projects/nscl_reproducability_challenge/data/test/images'
train_scene_json = '/Users/mark/Projects/nscl_reproducability_challenge/data/test/train_scenes.json'
train_question_json = '/Users/mark/Projects/nscl_reproducability_challenge/data/test/train_questions.json'

val_img_root = osp.abspath(osp.dirname(os.getcwd())) + '/data/CLEVR_v1.0/images/val'
val_scene_json = osp.abspath(osp.dirname(os.getcwd())) + '/data/CLEVR_v1.0/scenes/val/scenes.json'

dataset = build_clevr_dataset(train_img_root, train_scene_json, train_question_json)

visual_module = VisualModule()
img_feature, obj_feature = visual_module(dataset[0])

print('Image Feature shape:', img_feature.shape)
print('Box Feature shape:', obj_feature.shape)

print(obj_feature)

