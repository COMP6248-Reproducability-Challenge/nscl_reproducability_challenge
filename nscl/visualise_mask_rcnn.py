import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from nscl.datasets.clevr_dataset import build_clevr_dataset
from pycocotools import mask as mask_utils

img_root = os.path.abspath(os.getcwd()) + '/data/test/images'
scene_json = os.path.abspath(os.getcwd()) + '/data/test/train_scenes.json'
questions_json = os.path.abspath(os.getcwd()) + '/data/test/train_questions.json'

dataset = build_clevr_dataset(img_root, scene_json, questions_json)
img, question, scene = dataset[0]
boxes = [mask_utils.toBbox(i['mask']) for i in scene.detection]
fig, ax = plt.subplots(1)

ax.axis("off")
plt.title(f'{question.raw_question} {question.answer}')
ax.imshow(img.permute(1, 2, 0))
for box in boxes:
    p = patches.Rectangle((box[0], box[1]), box[2], box[3], edgecolor='r', facecolor='none')
    ax.add_patch(p)

plt.show()
