import os
import os.path as osp

import torch
import wandb
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split
from tqdm import tqdm

from nscl.datasets.clevr_dataset import build_clevr_dataset, build_clevr_dataloader
from nscl.datasets.clevr_definition import CLEVRDefinition
from nscl.models.loss.nscl_loss import QALoss, CESceneParsingLoss
from nscl.models.nscl_module import NSCLModule

wandb.init(project="nscl-reproduce-final-1")
save_interval = 5  # epoch

train_img_root = osp.abspath(os.getcwd()) + '/data/CLEVR_v1.0/images/train'
train_scene_json = osp.abspath(os.getcwd()) + '/data/CLEVR_v1.0/scenes/train/scenes.json'
train_question_json = osp.abspath(os.getcwd()) + '/data/CLEVR_v1.0/questions/CLEVR_train_questions.json'

device = "cuda:3" if torch.cuda.is_available() else "cpu"

# epoch, max_program_size, max_scene_size
curriculum_strategies = [
    # lesson 1
    (5, 4, 3),
    (5, 6, 3),
    (5, 8, 3),
    # # lesson 2
    # (10, 8, 4),
    # (10, 12, 4),
    # (10, 12, 5),
    # # lesson 3
    # (10, 12, 6),
    # (10, 16, 7),
    # (10, 20, 8),
    # (10, 22, 9),
    # (15, 25, 10)
]

batch_size = 64
num_workers = 0

model = NSCLModule(CLEVRDefinition.attribute_concept_map).to(device)
wandb.watch(model)
# model.load_state_dict(torch.load('nscl-5.pt', map_location=torch.device(device)))

scene_parsing_loss = CESceneParsingLoss(CLEVRDefinition.attribute_concept_map, reduction='sum')
qa_loss = QALoss(reduction='sum')

opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
# scheduler = StepLR(opt, step_size=5, gamma=0.1)

for i, (epoch, max_program_size, max_scene_size) in enumerate(curriculum_strategies):
    print(f'Curriculum strategy: {max_program_size}, {max_scene_size}')
    dataset = build_clevr_dataset(train_img_root, train_scene_json, train_question_json,
                                  max_program_size=max_program_size, max_scene_size=max_scene_size,
                                  gen_basic_scene_questions=True)
    train_dataset, val_dataset = random_split(dataset, [len(dataset) - len(dataset) // 4, len(dataset) // 4])
    train_loader = build_clevr_dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                          drop_last=False)
    val_loader = build_clevr_dataloader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                        drop_last=False)

    for e in range(1, epoch + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_loader), desc='train') as t:
            for idx, (images, questions, scenes) in enumerate(train_loader):
                opt.zero_grad()
                object_annotations, answers = model(images.to(device), questions, scenes)
                scene_loss = scene_parsing_loss(object_annotations, scenes)
                q_loss = qa_loss(questions, answers)
                total_loss = scene_loss + q_loss
                total_loss.backward()
                opt.step()

                epoch_loss = (epoch_loss * idx + total_loss.item()) / (idx + 1)
                wandb.log({"train_epoch_loss": epoch_loss, "train_qa_loss": q_loss.item(),
                           "train_scene_loss": scene_loss.item()})
                t.set_postfix(loss='{:05.3f}'.format(epoch_loss))
                t.update()

        model.eval()
        epoch_loss = 0
        with tqdm(total=len(val_loader), desc='val') as t:
            with torch.no_grad():
                for idx, (images, questions, scenes) in enumerate(val_loader):
                    object_annotations, answers = model(images.to(device), questions, scenes)
                    scene_loss = scene_parsing_loss(object_annotations, scenes)
                    q_loss = qa_loss(questions, answers)
                    total_loss = scene_loss + q_loss

                    epoch_loss = (epoch_loss * idx + total_loss.item()) / (idx + 1)
                    wandb.log({"val_epoch_loss": epoch_loss, "val_qa_loss": q_loss.item(),
                               "val_scene_loss": scene_loss.item()})
                    t.set_postfix(loss='{:05.3f}'.format(epoch_loss))
                    t.update()

        # scheduler.step()

        if e > 0 and e % save_interval == 0:
            torch.save(model.state_dict(), f'nscl-{i}-{e}.pt')
