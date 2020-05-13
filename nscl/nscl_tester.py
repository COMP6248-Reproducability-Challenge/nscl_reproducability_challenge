import os
import os.path as osp

from nscl.datasets.clevr_dataset import build_clevr_dataset, build_clevr_dataloader, CLEVRCurriculumSampler
from nscl.datasets.clevr_definition import CLEVRDefinition
from nscl.models.nscl_module import NSCLModule

train_img_root = osp.abspath(osp.dirname(osp.dirname(os.getcwd()))) + '/data/CLEVR_v1.0/images/train'
train_scene_json = osp.abspath(osp.dirname(osp.dirname(os.getcwd()))) + '/data/CLEVR_v1.0/scenes/train/scenes.json'
train_question_json = osp.abspath(
    osp.dirname(osp.dirname(os.getcwd()))) + '/data/CLEVR_v1.0/questions/CLEVR_train_questions.json'

batch_size = 100
num_workers = 4
dataset = build_clevr_dataset(train_img_root, train_scene_json, train_question_json)
curriculum_sampler = CLEVRCurriculumSampler(dataset, max_scene_size=5, max_program_size=5, max_data_size=1000)
data_loader = build_clevr_dataloader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                     drop_last=False, sampler=curriculum_sampler)

model = NSCLModule(CLEVRDefinition.attribute_concept_map)

for images, questions, scenes in data_loader:
    results = model(images, questions, scenes)
    for i, q in enumerate(questions):
        print(q.raw_question, q.answer)
    print(results)
    break
