import os
import os.path as osp

from nscl.datasets.clevr_dataset import build_clevr_dataset, build_clevr_dataloader
from nscl.datasets.clevr_definition import CLEVRDefinition
from nscl.models.nscl_module import NSCLModule

train_img_root = osp.abspath(osp.dirname(osp.dirname(os.getcwd()))) + '/data/CLEVR_v1.0/images/train'
train_scene_json = osp.abspath(osp.dirname(osp.dirname(os.getcwd()))) + '/data/CLEVR_v1.0/scenes/train/scenes.json'
train_question_json = osp.abspath(
    osp.dirname(osp.dirname(os.getcwd()))) + '/data/CLEVR_v1.0/questions/CLEVR_train_questions.json'

batch_size = 100
dataset = build_clevr_dataset(train_img_root, train_scene_json, train_question_json)
# <= 5 objects and programs in question
# curriculum_sampler = CLEVRCurriculumSampler(dataset, max_scene_size=3, max_program_size=3)
data_loader = build_clevr_dataloader(dataset, batch_size=100, num_workers=4, shuffle=False,
                                     drop_last=True)  # , sampler=curriculum_sampler)

model = NSCLModule(CLEVRDefinition.attribute_concept_map)

for images, questions, scenes in data_loader:
    for idx in range(batch_size):
        image, question, scene = images[idx], questions[idx], scenes[idx]
        if len(question.program) < 5:
            results = model(image.unsqueeze(0), [question], [scene])
            print(results)
    break
