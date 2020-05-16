import json
import os.path as osp

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import Sampler

from nscl.datasets.question import Question
from nscl.datasets.scene import Scene

__all__ = ['CLEVRDataset', 'build_clevr_dataset', 'build_clevr_dataloader', 'CLEVRCurriculumSampler']


class CLEVRDataset(Dataset):

    def __init__(self, img_root, scene_json, questions_json, img_transform=None):
        super().__init__()

        self.img_location = img_root
        print(f'loading scenes from: {scene_json}')
        self.raw_scenes = json.load(open(scene_json))['scenes']
        print(f'loading questions from: {questions_json}')
        self.raw_questions = json.load(open(questions_json))['questions']
        self.img_transform = img_transform

    def __getitem__(self, index):
        question = Question(self.raw_questions[index])
        scene = Scene(self.raw_scenes[question.img_index])
        try:
            img = self.img_transform(Image.open(osp.join(self.img_location, question.img_file)).convert('RGB'))
        except Exception as ex:
            print(f'Unable to load image {question.img_file}')
            img = None
        return img, question, scene

    def __len__(self):
        return len(self.raw_questions)


def build_clevr_dataset(img_root, scenes_json, questions_json, img_transform=None):
    # transform for resnet model
    if img_transform is None:
        image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return CLEVRDataset(img_root, scenes_json, questions_json, image_transform)
    else:
        return CLEVRDataset(img_root, scenes_json, questions_json, img_transform)


def build_clevr_dataloader(dataset, batch_size, num_workers, shuffle, drop_last, max_scene_size, max_program_size):
    unimplemented_operator = ['relate', 'relate_attribute_equal', 'count_less', 'count_greater', 'count_equal']

    def clevr_collate(batch):
        img_batch = []
        questions = []
        scenes = []
        for img, question, scene in batch:
            operators = [p.operator for p in question.program]
            intersect = list(set(unimplemented_operator) & set(operators))
            if len(intersect) > 0 or img is None:
                continue
            if len(scene.objects) <= max_scene_size and len(question.program) <= max_program_size:
                continue
            img_batch.append(img)
            questions.append(question)
            scenes.append(scene)
        return default_collate(img_batch), questions, scenes

    return DataLoader(dataset, collate_fn=clevr_collate, num_workers=num_workers, batch_size=batch_size,
                      shuffle=shuffle, drop_last=drop_last)


class CLEVRCurriculumSampler(Sampler):

    def __init__(self, data_source, max_scene_size, max_program_size, max_data_size=None):
        super().__init__(data_source)
        self.data_source = data_source
        self.max_scene_size = max_scene_size
        self.max_program_size = max_program_size
        self.max_data_size = max_data_size
        self.indices = []
        self.count = 0

        unimplemented_operator = ['relate', 'relate_attribute_equal', 'count_less']
        print('Preparing curriculum sampler....')
        for (index, data) in enumerate(self.data_source):
            img, question, scene = data

            operators = [p.operator for p in question.program]
            intersect = list(set(unimplemented_operator) & set(operators))
            if len(intersect) > 0:
                continue

            if len(scene.objects) <= max_scene_size and len(question.program) <= max_program_size:
                self.indices.append(index)

            self.count += 1
            if self.max_data_size is not None and self.count >= self.max_data_size:
                break

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
