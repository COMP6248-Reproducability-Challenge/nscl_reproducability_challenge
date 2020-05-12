import os
import os.path as osp

from nscl.datasets.clevr_dataset import build_clevr_dataset, build_clevr_dataloader
from nscl.datasets.clevr_definition import CLEVRDefinition
from nscl.models.nscl_module import NSCLModule

train_img_root = '/Users/mark/Projects/nscl_reproducability_challenge/data/test/images'
train_scene_json = '/Users/mark/Projects/nscl_reproducability_challenge/data/test/train_scenes.json'
train_question_json = '/Users/mark/Projects/nscl_reproducability_challenge/data/test/train_questions.json'

val_img_root = osp.abspath(osp.dirname(os.getcwd())) + '/data/CLEVR_v1.0/images/val'
val_scene_json = osp.abspath(osp.dirname(os.getcwd())) + '/data/CLEVR_v1.0/scenes/val/scenes.json'

dataset = build_clevr_dataset(train_img_root, train_scene_json, train_question_json)
data_loader = build_clevr_dataloader(dataset, batch_size=10, shuffle=True, drop_last=False)

model = NSCLModule(CLEVRDefinition.attribute_concept_map)

imgs, questions, scenes = next(iter(data_loader))
results = model(imgs, questions, scenes)
print(results)
