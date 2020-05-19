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

    def __init__(self, img_root, scene_json, questions_json, max_program_size=None, max_scene_size=None, img_transform=None):
        super().__init__()

        self.img_location = img_root
        print(f'loading scenes from: {scene_json}')
        self.raw_scenes = json.load(open(scene_json))['scenes']
        print(f'loading questions from: {questions_json}')
        self.raw_questions = json.load(open(questions_json))['questions']
        self.img_transform = img_transform
        self.questions = [Question(q) for q in self.raw_questions]
        self.scenes = [Scene(s) for s in self.raw_scenes]
        if max_program_size is not None and max_program_size is not None:
            self.questions = CLEVRDataset.filter_questions(self.questions, self.scenes, max_program_size, max_scene_size)

    def __getitem__(self, index):
        question = self.questions[index]
        scene = self.scenes[question.img_index]
        try:
            img = self.img_transform(Image.open(osp.join(self.img_location, question.img_file)).convert('RGB'))
        except Exception as ex:
            print(f'Unable to load image {question.img_file}')
            img = None
        return img, question, scene

    def __len__(self):
        return len(self.questions)

    @staticmethod
    def filter_questions(questions, scenes, max_program_size, max_scene_size):
        program_filtered_questions = list(filter(None, [q if len(q.program) <= max_program_size else None for q in questions]))
        scene_filtered_questions = list(filter(None, [q if len(scenes[q.img_index].objects) <= max_scene_size else None for q in program_filtered_questions]))
        
        filtered_questions = []
        unimplemented_operator = ['relate', 'relate_attribute_equal', 'count_less', 'count_greater', 'count_equal']
        for q in scene_filtered_questions:
            operators = [p.operator for p in q.program]
            intersect = list(set(unimplemented_operator) & set(operators))
            if len(intersect) == 0:
                filtered_questions.append(q)
        return filtered_questions


def build_clevr_dataset(img_root, scenes_json, questions_json, max_program_size=None, max_scene_size=None, img_transform=None):
    # transform for resnet model
    if img_transform is None:
        image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return CLEVRDataset(img_root, scenes_json, questions_json, max_program_size, max_scene_size, image_transform)
    else:
        return CLEVRDataset(img_root, scenes_json, questions_json, max_program_size, max_scene_size, img_transform)


def build_clevr_dataloader(dataset, batch_size, num_workers, shuffle, drop_last, sampler=None):
    def clevr_collate(batch):
        img_batch = []
        questions = []
        scenes = []
        for _batch in batch:
            img_batch.append(_batch[0])
            questions.append(_batch[1])
            scenes.append(_batch[2])
        return default_collate(img_batch), questions, scenes

    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last, collate_fn=clevr_collate, sampler=sampler)
