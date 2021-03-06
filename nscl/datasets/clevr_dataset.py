import json
import os.path as osp

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from nscl.datasets.question import Question
from nscl.datasets.scene import Scene
from nscl.datasets.clevr_definition import CLEVRDefinition
import copy
import random

__all__ = ['CLEVRDataset', 'build_clevr_dataset', 'build_clevr_dataloader']


class CLEVRDataset(Dataset):

    def __init__(self, img_root, scene_json, questions_json, max_program_size=None, max_scene_size=None,
                 img_transform=None, gen_basic_scene_questions=False, num_questions_per_scene=20,
                 filter_non_equal_obj=False, count_question_only=False):
        super().__init__()

        self.img_location = img_root
        print(f'loading scenes from: {scene_json}')
        self.raw_scenes = json.load(open(scene_json))['scenes']
        print(f'loading questions from: {questions_json}')
        self.raw_questions = json.load(open(questions_json))['questions']
        self.img_transform = img_transform
        self.questions = [Question(q) for q in self.raw_questions]
        self.scenes = [Scene(s) for s in self.raw_scenes]
        if max_program_size is not None and max_scene_size is not None:
            print(f'Filtering dataset with max_program_size: {max_program_size} and max_scene_size: {max_scene_size}')
            self.questions = CLEVRDataset.filter_questions(self.questions, self.scenes, max_program_size,
                                                           max_scene_size)

        if filter_non_equal_obj:
            self.questions = CLEVRDataset.filter_non_equal_objects(self.questions, self.scenes)

        if count_question_only:
            count_questions = [CLEVRDataset.generate_count_only_questions(s, num_questions_per_scene) for s in
                               self.scenes if len(s.objects) <= max_scene_size]
            self.questions = [q for questions in count_questions for q in questions]

        basic_scene_questions = []
        if gen_basic_scene_questions and max_scene_size is not None and not count_question_only:
            print(f'Generating additional questions...')
            basic_scene_questions = [CLEVRDataset.generate_basic_scene_questions(s, num_questions_per_scene) for s in
                                     self.scenes if len(s.objects) <= max_scene_size]
            basic_scene_questions = [q for questions in basic_scene_questions for q in questions]

        self.questions.extend(basic_scene_questions)
        print('Dataset preparation completed...')

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
        program_filtered_questions = list(
            filter(None, [q if len(q.program) <= max_program_size else None for q in questions]))
        scene_filtered_questions = list(filter(None,
                                               [q if len(scenes[q.img_index].objects) <= max_scene_size else None for q
                                                in program_filtered_questions]))

        filtered_questions = []
        unimplemented_operator = ['relate', 'relate_attribute_equal', 'count_less', 'count_greater', 'count_equal']
        for q in scene_filtered_questions:
            operators = [p.operator for p in q.program]
            intersect = list(set(unimplemented_operator) & set(operators))
            if len(intersect) == 0:
                filtered_questions.append(q)
        return filtered_questions

    @staticmethod
    def filter_non_equal_objects(questions, scenes):
        return [q for q in questions if len(scenes[q.img_index].objects) == len(scenes[q.img_index].rearranged_objects)]

    @staticmethod
    def generate_similar_questions(q, scene):
        question_type = q.program[-1].operator
        if (question_type == 'count' or question_type == 'exist') and len(q.program) == 3:
            return CLEVRDataset.generate_similar_count_or_exist_questions(q, scene)
        elif question_type == 'query' and len(q.program) == 4:
            return CLEVRDataset.generate_similar_query_questions(q, scene)
        else:
            return []

    @staticmethod
    def generate_count_only_questions(scene, num_questions):
        basic_questions = []
        for c in CLEVRDefinition.get_all_concepts():
            count_question = Question.gen_count_question(c)
            count_question.img_index = scene.img_index
            count_question.img_file = scene.img_filename
            count_answer = str(len(CLEVRDataset.filter_objects_by_concept(scene, c)))
            count_question.answer = count_answer
            count_question.answer_tensor = Question.get_answer_tensor(count_answer)
            count_question.question_type = Question.get_question_type(count_answer)

            basic_questions.append(count_question)

        random.shuffle(basic_questions)
        return basic_questions[0:min(num_questions, len(basic_questions))]

    @staticmethod
    def generate_basic_scene_questions(scene, num_questions):
        basic_questions = []
        for c in CLEVRDefinition.get_all_concepts():
            count_question = Question.gen_count_question(c)
            count_question.img_index = scene.img_index
            count_question.img_file = scene.img_filename
            count_answer = str(len(CLEVRDataset.filter_objects_by_concept(scene, c)))
            count_question.answer = count_answer
            count_question.answer_tensor = Question.get_answer_tensor(count_answer)
            count_question.question_type = Question.get_question_type(count_answer)

            exist_question = Question.gen_exist_question(c)
            exist_question.img_index = scene.img_index
            exist_question.img_file = scene.img_filename
            exist_answer = 'yes' if len(CLEVRDataset.filter_objects_by_concept(scene, c)) > 0 else 'no'
            exist_question.answer = exist_answer
            exist_question.answer_tensor = Question.get_answer_tensor(exist_answer)
            count_question.question_type = Question.get_question_type(count_answer)

            basic_questions.extend([count_question, exist_question])

        random.shuffle(basic_questions)
        return basic_questions[0:min(num_questions, len(basic_questions))]

    @staticmethod
    def generate_similar_count_or_exist_questions(q, scene):
        new_questions = []
        for c in CLEVRDefinition.get_all_concepts():
            new_question = CLEVRDataset.modify_filter(q, c)
            if q.program[-1].operator == 'count':
                new_answer = str(len(CLEVRDataset.filter_objects_by_concept(scene, c)))
            else:
                new_answer = 'yes' if len(CLEVRDataset.filter_objects_by_concept(scene, c)) > 0 else 'no'
            new_question.answer = new_answer
            new_question.answer_tensor = Question.get_answer_tensor(new_answer)
            new_question.synthetic = True
            new_questions.append(new_question)
        return new_questions

    @staticmethod
    def generate_similar_query_questions(q, scene):
        new_questions = []
        filter_program = [p for p in q.program if p.operator == 'filter'][0]
        filter_attr, filter_concept = filter_program.attribute, filter_program.concept
        object_to_query = CLEVRDataset.filter_objects_by_concept(scene, filter_concept)[0]
        for attr in CLEVRDefinition.get_all_attributes():
            if attr == q.program[-1].attribute or attr == filter_attr: continue
            new_question = copy.deepcopy(q)
            new_question.raw_question = new_question.raw_question.replace(q.program[-1].attribute, attr)
            new_question.program[-1].attribute = attr
            new_question.answer = getattr(object_to_query, attr)
            new_question.answer_tensor = Question.get_answer_tensor(new_question.answer)
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


def build_clevr_dataset(img_root, scenes_json, questions_json, max_program_size=None, max_scene_size=None,
                        img_transform=None, gen_basic_scene_questions=False, filter_non_equal_obj=False,
                        count_question_only=False, num_questions_per_scene=20):
    # transform for resnet model
    if img_transform is None:
        img_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    return CLEVRDataset(img_root, scenes_json, questions_json, max_program_size, max_scene_size, img_transform,
                        gen_basic_scene_questions, filter_non_equal_obj=filter_non_equal_obj,
                        count_question_only=count_question_only, num_questions_per_scene=num_questions_per_scene)


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

    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last,
                      collate_fn=clevr_collate, sampler=sampler)
