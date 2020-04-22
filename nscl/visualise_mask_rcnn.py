import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from nscl.datasets.clevr_dataset import build_clevr_dataset
from pycocotools import mask as mask_utils


def label(xy, text):
    x = xy[0] + 10
    y = xy[1] - 0.15  # shift y-value for label so that it's below the artist
    plt.text(x, y, text, ha="center", family='sans-serif', size=14)


img_root = os.path.abspath(os.getcwd()) + '/data/test/images'
scene_json = os.path.abspath(os.getcwd()) + '/data/test/train_scenes.json'
questions_json = os.path.abspath(os.getcwd()) + '/data/test/train_questions.json'

dataset = build_clevr_dataset(img_root, scene_json, questions_json)
img, question, scene = dataset[0]
boxes = [mask_utils.toBbox(i['mask']) for i in scene.detection]
fig, ax = plt.subplots(1)

plt.title(f'{question.raw_question} {question.answer}')
ax.imshow(img.permute(1, 2, 0))
for box in boxes:
    x, y, width, height = box[0], box[1], box[2], box[3]
    p = patches.Rectangle((x, y), width, height, edgecolor='r', facecolor='none', label='Label')
    obj = next((obj for obj in scene.objects if x < obj.coordinates[0] < x + width
                and y < obj.coordinates[1] < y + height), None)
    ax.add_patch(p)
    label((x, y), obj.shape)

plt.show()
