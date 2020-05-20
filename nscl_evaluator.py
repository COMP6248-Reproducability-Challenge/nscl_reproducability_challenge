import os
import os.path as osp

import torch
from torch import nn
from tqdm import tqdm

from nscl.datasets.clevr_dataset import build_clevr_dataset, build_clevr_dataloader
from nscl.datasets.clevr_definition import CLEVRDefinition, QuestionTypes
from nscl.models.nscl_module import NSCLModule

batch_size = 64
num_workers = 0
test_img_root = osp.abspath(os.getcwd()) + '/data/CLEVR_v1.0/images/val'
test_scene_json = osp.abspath(os.getcwd()) + '/data/CLEVR_v1.0/scenes/val/scenes.json'
test_question_json = osp.abspath(os.getcwd()) + '/data/CLEVR_v1.0/questions/CLEVR_val_questions.json'

test_dataset = build_clevr_dataset(test_img_root, test_scene_json, test_question_json, max_scene_size=5,
                                   max_program_size=5)
test_loader = build_clevr_dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                     drop_last=False)

mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()
ce_loss = nn.CrossEntropyLoss()

device = "cuda:1" if torch.cuda.is_available() else "cpu"
model = NSCLModule(CLEVRDefinition.attribute_concept_map).to(device)
model.load_state_dict(torch.load('nscl.pt', map_location=torch.device(device)))
model.eval()
correct = 0
count = 0
with tqdm(total=len(test_loader), desc='test') as t:
    with torch.no_grad():
        epoch_loss = 0
        for idx, (images, questions, scenes) in enumerate(test_loader):
            results = model(images.to(device), questions, scenes)
            total_loss = torch.tensor(0.0, dtype=torch.float)
            for i, q in enumerate(questions):
                predicted_answer = results[i]
                true_answer = q.answer_tensor.to(device)
                if q.question_type == QuestionTypes.COUNT:
                    loss = mse_loss(predicted_answer, true_answer)
                    # print(q.raw_question)
                    # print('True Answer', q.answer)
                    # print('Predicted Answer', int(round(predicted_answer.item())))
                    ans = int(round(predicted_answer.item()))
                    ans_t = int(q.answer)
                    if ans == ans_t:
                        correct += 1
                elif q.question_type == QuestionTypes.BOOLEAN:
                    predicted_answer = torch.stack(
                        [predicted_answer, torch.tensor(1 - predicted_answer.item(), device=device)])
                    loss = bce_loss(predicted_answer, true_answer)
                    # print(q.raw_question)
                    # print('True Answer', q.answer)
                    ans = 'yes' if int(torch.argmax(predicted_answer).item()) == 0 else 'no'
                    ans_t = q.answer
                    # print('Predicted Answer', ans)
                    if ans == ans_t:
                        correct += 1
                else:
                    predicted_answer = predicted_answer.unsqueeze(0)
                    loss = ce_loss(predicted_answer, true_answer)
                    # print(q.raw_question)
                    # print('True Answer', q.answer)
                    # print('True Answer id', q.answer_tensor.item())
                    ans = int(torch.argmax(predicted_answer).item())
                    ans_t = int(q.answer_tensor.item())
                    # print('Predicted Answer', ans)
                    if ans == ans_t:
                        correct += 1

                # print()
                count += 1
                total_loss = total_loss + loss

            epoch_loss = (epoch_loss * idx + total_loss.item()) / (idx + 1)
            t.set_postfix(loss='{:05.3f}'.format(epoch_loss))
            t.update()

print('correct', correct)
print('count', count)
print('Test Accuracy', correct / count)
