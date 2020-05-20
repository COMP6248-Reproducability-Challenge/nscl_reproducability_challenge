import os
import os.path as osp

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split
from tqdm import tqdm

from nscl.datasets.clevr_dataset import build_clevr_dataset, build_clevr_dataloader
from nscl.datasets.clevr_definition import CLEVRDefinition, QuestionTypes
from nscl.models.nscl_module import NSCLModule

import wandb

wandb.init(project="nscl-reproduce")
save_interval = 10  # epoch

train_img_root = osp.abspath(os.getcwd()) + '/data/CLEVR_v1.0/images/train'
train_scene_json = osp.abspath(os.getcwd()) + '/data/CLEVR_v1.0/scenes/train/scenes.json'
train_question_json = osp.abspath(os.getcwd()) + '/data/CLEVR_v1.0/questions/CLEVR_train_questions.json'

batch_size = 64
num_workers = 0
dataset = build_clevr_dataset(train_img_root, train_scene_json, train_question_json, max_scene_size=5,
                              max_program_size=5)
train_dataset, val_dataset = random_split(dataset, [len(dataset) - len(dataset) // 3, len(dataset) // 3])

train_loader = build_clevr_dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                      drop_last=False)
val_loader = build_clevr_dataloader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                    drop_last=False)

device = "cuda:1" if torch.cuda.is_available() else "cpu"
epoch = 100
model = NSCLModule(CLEVRDefinition.attribute_concept_map).to(device)

wandb.watch(model)

# model.load_state_dict(torch.load('nscl.pt', map_location=torch.device(device)))
mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()
ce_loss = nn.CrossEntropyLoss()

opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
scheduler = StepLR(opt, step_size=50, gamma=0.1)

for e in range(epoch):
    model.train()
    with tqdm(total=len(train_loader), desc='train') as t:
        epoch_loss = 0
        for idx, (images, questions, scenes) in enumerate(train_loader):
            opt.zero_grad()
            results = model(images.to(device), questions, scenes)
            total_loss = torch.tensor(0.0, requires_grad=True, dtype=torch.float, device=device)
            for i, q in enumerate(questions):
                predicted_answer = results[i]
                true_answer = q.answer_tensor.to(device)
                if q.question_type == QuestionTypes.COUNT:
                    loss = mse_loss(predicted_answer, true_answer)
                elif q.question_type == QuestionTypes.BOOLEAN:
                    predicted_answer = torch.stack(
                        [predicted_answer,
                         torch.tensor(1 - predicted_answer.item(), device=device, requires_grad=True)])
                    loss = bce_loss(predicted_answer, true_answer)
                else:
                    predicted_answer = predicted_answer.unsqueeze(0)
                    loss = ce_loss(predicted_answer, true_answer)

                total_loss = total_loss + loss

            total_loss.backward()
            opt.step()
            epoch_loss = (epoch_loss * idx + total_loss.item()) / (idx + 1)
            wandb.log({"epoch_loss": epoch_loss})
            t.set_postfix(loss='{:05.3f}'.format(epoch_loss))
            t.update()

    model.eval()
    with tqdm(total=len(val_loader), desc='val') as t:
        with torch.no_grad():
            epoch_loss = 0
            for idx, (images, questions, scenes) in enumerate(val_loader):
                results = model(images.to(device), questions, scenes)
                total_loss = torch.tensor(0.0, requires_grad=True, dtype=torch.float, device=device)
                for i, q in enumerate(questions):
                    predicted_answer = results[i]
                    true_answer = q.answer_tensor.to(device)
                    if q.question_type == QuestionTypes.COUNT:
                        loss = mse_loss(predicted_answer, true_answer)
                    elif q.question_type == QuestionTypes.BOOLEAN:
                        predicted_answer = torch.stack(
                            [predicted_answer, torch.tensor(1 - predicted_answer.item(), device=device)])
                        loss = bce_loss(predicted_answer, true_answer)
                    else:
                        predicted_answer = predicted_answer.unsqueeze(0)
                        loss = ce_loss(predicted_answer, true_answer)

                    total_loss = total_loss + loss

                epoch_loss = (epoch_loss * idx + total_loss.item()) / (idx + 1)
                t.set_postfix(loss='{:05.3f}'.format(epoch_loss))
                t.update()

    scheduler.step()

    if e % save_interval == 0:
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, f'nscl-{e}.pt'))
