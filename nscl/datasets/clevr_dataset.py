import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
import json
from question import Question
from scene import Scene
from PIL import Image
import os.path as osp
import torchvision.transforms as transforms

class CLEVRDataset(Dataset):

    def __init__(self, img_root, scene_json, questions_json, img_transform = None):
        super().__init__()

        self.img_location = img_root
        self.raw_scenes = json.load(open(scene_json))['scenes']
        self.raw_questions = json.load(open(questions_json))['questions']
        self.img_transform = img_transform

    def __getitem__(self, index):
        question = Question(self.raw_questions[index])
        scene = Scene(self.raw_scenes[question.img_index])
        img = self.img_transform(Image.open(osp.join(self.img_location, question.img_file)).convert('RGB'))
        sample = dict()
        sample['img'] = img
        sample['question'] = self.raw_questions[index]
        # sample['scene'] = scene
        return sample

    def __len__(self):
        return len(self.raw_questions)

def build_clevr_dataset(img_root, scenes_json, questions_json):
    img_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = CLEVRDataset(img_root, scenes_json, questions_json, img_transform,)
    return dataset