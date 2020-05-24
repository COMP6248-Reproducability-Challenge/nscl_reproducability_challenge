import json
import os.path as osp

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import Sampler
from nscl.datasets.question import Question
from nscl.datasets.scene import Scene
from nscl.datasets.clevr_definition import CLEVRDefinition
import copy

__all__ = ['CLEVRDataset', 'build_clevr_dataset', 'build_clevr_dataloader', 'CLEVRCurriculumSampler']


class CLEVRDataset(Dataset):

    def __init__(self, img_root, scene_json, questions_json, max_program_size=None, max_scene_size=None, img_transform=None, gen_similar_questions=False):
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
        
        if gen_similar_questions:
            new_questions = [CLEVRDataset.generate_similar_questions(q, self.scenes[q.img_index]) for q in self.questions if len(q.program) > 2]
            new_questions = [q for questions in new_questions for q in questions]
            self.questions = self.questions + new_questions

    def __getitem__(self, index):
        question = self.questions[index]
        scene = self.scenes[question.img_index]
        try:
            img = self.img_transform(Image.open(osp.join(self.img_location, question.img_file)).convert('RGB'))
        except Exception as ex:
            print(f'Unable to load image {question.img_file} {ex}')
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

    @staticmethod
    def generate_similar_questions( q, scene):
        question_type = q.program[-1].operator
        if question_type == 'count' or question_type == 'exist':
            return CLEVRDataset.generate_similar_count_or_exist_questions(q, scene)
        else:
            return []


    @staticmethod
    def generate_similar_count_or_exist_questions(q, scene):
        new_questions = []
        for c in CLEVRDefinition.get_all_concepts():
            new_question = CLEVRDataset.modify_filter(q, c)
            if q.program[-1].operator == 'count': new_answer = str(len(CLEVRDataset.filter_objects_by_concept(scene, c)))
            else                                : new_answer = 'yes' if len(CLEVRDataset.filter_objects_by_concept(scene, c)) > 0 else 'no'
            new_question.answer = new_answer
            new_question.answer_tensor = Question.get_answer_tensor(new_answer)
            new_question.synthetic = True
            new_questions.append(new_question)
        return new_questions

    @staticmethod
    def modify_filter(q, new_concept):
        new_question = copy.deepcopy(q)
        filter_program = [p for p in new_question.program if p.operator == 'filter'][0]
        new_question.raw_question = new_question.raw_question.replace(filter_program.concept, new_concept)
        filter_program.concept = new_concept
        filter_program.attribute = CLEVRDefinition.concept_attribute_map[new_concept]
        return new_question

    @staticmethod
    def filter_objects_by_concept(scene, concept):
        return [obj for obj in scene.objects if getattr(obj, CLEVRDefinition.concept_attribute_map[concept]) == concept]

def build_clevr_dataset(img_root, scenes_json, questions_json, max_program_size=None, max_scene_size=None, img_transform=None, gen_similar_questions=False):
    # transform for resnet model
    if img_transform is None:
        img_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    return CLEVRDataset(img_root, scenes_json, questions_json, max_program_size, max_scene_size, img_transform, gen_similar_questions)


def build_clevr_dataloader(dataset, batch_size, num_workers, shuffle, drop_last, sampler=None):
    def clevr_collate(batch):
        img_batch = []
        questions = []
        scenes = []
        for img, question, scene in batch:
            if img is None:
                continue
            img_batch.append(img)
            questions.append(question)
            scenes.append(scene)
        return default_collate(img_batch), questions, scenes

    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last, collate_fn=clevr_collate, sampler=sampler)
