import os
import os.path as osp

import torch

from nscl.datasets.clevr_dataset import build_clevr_dataset, build_clevr_dataloader
from nscl.models.visual.visual_module import VisualModule

train_img_root = osp.abspath(os.getcwd()) + '/data/CLEVR_v1.0/images/train'
train_scene_json = osp.abspath(os.getcwd()) + '/data/CLEVR_v1.0/scenes/train/scenes.json'
train_question_json = osp.abspath(os.getcwd()) + '/data/CLEVR_v1.0/questions/CLEVR_train_questions.json'

device = "cuda:0" if torch.cuda.is_available() else "cpu"

batch_size = 64
num_workers = 0

train_dataset = build_clevr_dataset(train_img_root, train_scene_json, train_question_json, max_scene_size=5,
                                    max_program_size=5)
train_loader = build_clevr_dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                      drop_last=False)

visual_module = VisualModule()

obj_features = visual_module(imgs, scenes)

