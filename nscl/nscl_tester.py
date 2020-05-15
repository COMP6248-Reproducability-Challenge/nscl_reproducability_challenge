import os
import os.path as osp

import torch
from torch import nn
from torch import optim

from nscl.datasets.clevr_dataset import build_clevr_dataset, build_clevr_dataloader, CLEVRCurriculumSampler
from nscl.datasets.clevr_definition import CLEVRDefinition, QuestionTypes
from nscl.models.nscl_module import NSCLModule

train_img_root = osp.abspath(osp.dirname(osp.dirname(os.getcwd()))) + '/data/CLEVR_v1.0/images/train'
train_scene_json = osp.abspath(osp.dirname(osp.dirname(os.getcwd()))) + '/data/CLEVR_v1.0/scenes/train/scenes.json'
train_question_json = osp.abspath(
    osp.dirname(osp.dirname(os.getcwd()))) + '/data/CLEVR_v1.0/questions/CLEVR_train_questions.json'

batch_size = 100
num_workers = 4
dataset = build_clevr_dataset(train_img_root, train_scene_json, train_question_json)
curriculum_sampler = CLEVRCurriculumSampler(dataset, max_scene_size=5, max_program_size=5, max_data_size=500)
data_loader = build_clevr_dataloader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                     drop_last=False, sampler=curriculum_sampler)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
epoch = 10
model = NSCLModule(CLEVRDefinition.attribute_concept_map).to(device)
mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()
ce_loss = nn.CrossEntropyLoss()

opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)

for e in range(epoch):
    model.train()
    print('running...')
    epoch_loss = 0
    for idx, (images, questions, scenes) in enumerate(data_loader):
        opt.zero_grad()
        results = model(images.to(device), questions, scenes)
        total_loss = torch.tensor(0.0, requires_grad=True, dtype=torch.float)
        for i, q in enumerate(questions):
            if q.question_type == QuestionTypes.COUNT:
                loss = mse_loss(results[i], q.answer_tensor)
            elif q.question_type == QuestionTypes.BOOLEAN:
                loss = bce_loss(results[i], q.answer_tensor)
            else:
                loss = bce_loss(results[i], q.answer_tensor)

            total_loss = total_loss + loss

        total_loss.backward()
        opt.step()
        epoch_loss = (epoch_loss * idx + total_loss.item()) / (idx + 1)
        print(f'epoch {e} loss', epoch_loss)


