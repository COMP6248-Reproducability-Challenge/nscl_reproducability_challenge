import os
import os.path as osp
from collections import defaultdict

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
                                   max_program_size=4, count_question_only=True, num_questions_per_scene=1)
test_loader = build_clevr_dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                     drop_last=False)

device = "cuda:1" if torch.cuda.is_available() else "cpu"
model = NSCLModule(CLEVRDefinition.attribute_concept_map).to(device)
model.load_state_dict(torch.load('nscl_final.weights', map_location=torch.device(device)))
model.eval()

attribute_correct = defaultdict(int)
attribute_count = defaultdict(int)

with tqdm(total=len(test_loader), desc='test') as t:
    with torch.no_grad():
        for idx, (images, questions, scenes) in enumerate(test_loader):
            _, results = model(images.to(device), questions, scenes)

            for i, q in enumerate(questions):
                attr = q.program[1].attribute
                if q.question_type == QuestionTypes.COUNT:
                    predicted_answer = int(round(results[i].item()))
                    true_answer = int(q.answer)
                if true_answer == predicted_answer:
                    attribute_correct[attr] += 1
                attribute_count[attr] += 1

            # t.set_postfix(acc='{:.3f}'.format(correct / count))
            t.update()

total_correct = 0
total_count = 0
for attr in CLEVRDefinition.attribute_concept_map.keys():
    print(f'Attribute: {attr}')
    print('Correct', attribute_correct[attr])
    print('Count', attribute_count[attr])
    print('Accuracy', attribute_correct[attr] / attribute_count[attr])
    total_correct += attribute_correct[attr]
    total_count += attribute_count[attr]
    print()

print('Total Correct', total_correct)
print('Total Count', total_count)
print('Total Accuracy', total_correct / total_count)
