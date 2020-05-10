import os
import os.path as osp

from nscl.datasets.clevr_dataset import build_clevr_dataset, build_clevr_dataloader
from nscl.models.visual.visual_module import VisualModule

train_img_root = '/Users/mark/Projects/nscl_reproducability_challenge/data/test/images'
train_scene_json = '/Users/mark/Projects/nscl_reproducability_challenge/data/test/train_scenes.json'
train_question_json = '/Users/mark/Projects/nscl_reproducability_challenge/data/test/train_questions.json'

val_img_root = osp.abspath(osp.dirname(os.getcwd())) + '/data/CLEVR_v1.0/images/val'
val_scene_json = osp.abspath(osp.dirname(os.getcwd())) + '/data/CLEVR_v1.0/scenes/val/scenes.json'

dataset = build_clevr_dataset(train_img_root, train_scene_json, train_question_json)
data_loader = build_clevr_dataloader(dataset, batch_size=10, shuffle=True, drop_last=False)

visual_module = VisualModule()

imgs, questions, scenes = next(iter(data_loader))
obj_features = visual_module(imgs, questions, scenes)

print('Object Features:\n', len(obj_features))