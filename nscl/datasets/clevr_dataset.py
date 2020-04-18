import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
import json
from nscl.datasets.question import Question
from nscl.datasets.scene import Scene
from PIL import Image
import os.path as osp
import torchvision.transforms as transforms

__all__ = ['CLEVRDataset', 'build_clevr_dataset', 'build_clevr_dataloader']


class CLEVRDataset(Dataset):

    def __init__(self, img_root, scene_json, questions_json, img_transform=None):
        super().__init__()

        self.img_location = img_root
        self.raw_scenes = json.load(open(scene_json))['scenes']
        self.raw_questions = json.load(open(questions_json))['questions']
        self.img_transform = img_transform

    def __getitem__(self, index):
        question = Question(self.raw_questions[index])
        scene = Scene(self.raw_scenes[question.img_index])
        img = self.img_transform(Image.open(osp.join(self.img_location, question.img_file)).convert('RGB'))
        return img, question, scene

    def __len__(self):
        return len(self.raw_questions)

def build_clevr_dataset(img_root, scenes_json, questions_json):
    img_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = CLEVRDataset(img_root, scenes_json, questions_json, img_transform)
    return dataset


def build_clevr_dataloader(dataset, batch_size, shuffle, drop_last):
    def clevr_collate(batch):
        img_batch = []
        questions = []
        scenes = []
        for _batch in batch:
            img_batch.append(_batch[0])
            questions.append(_batch[1])
            scenes.append(_batch[2])
        return default_collate(img_batch), questions, scenes

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=clevr_collate)
