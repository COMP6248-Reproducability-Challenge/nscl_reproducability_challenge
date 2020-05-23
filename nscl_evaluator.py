import os
import os.path as osp

import torch
from tqdm import tqdm

from nscl.datasets.clevr_dataset import build_clevr_dataset, build_clevr_dataloader
from nscl.datasets.clevr_definition import CLEVRDefinition, QuestionTypes
from nscl.models.nscl_module import NSCLModule

batch_size = 64
num_workers = 0
test_img_root = osp.abspath(os.getcwd()) + '/data/CLEVR_v1.0/images/val'
test_scene_json = osp.abspath(os.getcwd()) + '/data/CLEVR_v1.0/scenes/val/scenes.json'
test_question_json = osp.abspath(os.getcwd()) + '/data/CLEVR_v1.0/questions/CLEVR_val_questions.json'

test_dataset = build_clevr_dataset(test_img_root, test_scene_json, test_question_json, max_scene_size=3,
                                   max_program_size=3)
test_loader = build_clevr_dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                     drop_last=False)

device = "cuda:1" if torch.cuda.is_available() else "cpu"
model = NSCLModule(CLEVRDefinition.attribute_concept_map).to(device)
model.load_state_dict(torch.load('nscl-5.pt', map_location=torch.device(device)))
model.eval()

correct = 0
count = 0
with tqdm(total=len(test_loader), desc='test') as t:
    with torch.no_grad():
        for idx, (images, questions, scenes) in enumerate(test_loader):
            _, results = model(images.to(device), questions, scenes)

            for i, q in enumerate(questions):
                if q.question_type == QuestionTypes.COUNT:
                    predicted_answer = int(round(results[i].item()))
                    true_answer = int(q.answer)
                elif q.question_type == QuestionTypes.BOOLEAN:
                    predicted_answer = 'yes' if int(torch.argmax(results[i]).item()) == 0 else 'no'
                    true_answer = q.answer
                else:
                    predicted_answer = int(torch.argmax(results[i]).item())
                    true_answer = int(q.answer_tensor.item())

                if true_answer == predicted_answer:
                    correct += 1
                count += 1

            t.set_postfix(acc='{:.3f}'.format(correct / count))
            t.update()

print('correct', correct)
print('count', count)
print('Test Accuracy', correct / count)
