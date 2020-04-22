import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.models import resnet34

from nscl.clevr_object_dataset import CLEVRObjectDataset

# fix random seed
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
torch.manual_seed(2019)
torch.cuda.manual_seed(2019)
torch.cuda.manual_seed_all(2019)
torch.backends.cudnn.deterministic = True

# average obj size [56.586 62.133]
img_preprocess = transforms.Compose([
    # transforms.Grayscale(),
    transforms.Resize(60),
    transforms.CenterCrop(60),
    transforms.ToTensor()
])

train_img_root = os.path.abspath(osp.dirname(os.getcwd())) + '/data/CLEVR_v1.0/images/train'
train_scene_json = os.path.abspath(osp.dirname(os.getcwd())) + '/data/CLEVR_v1.0/scenes/train/scenes.json'
val_img_root = os.path.abspath(osp.dirname(os.getcwd())) + '/data/CLEVR_v1.0/images/val'
val_scene_json = os.path.abspath(osp.dirname(os.getcwd())) + '/data/CLEVR_v1.0/scenes/val/scenes.json'

print('Preparing Dataset')

train_dataset = CLEVRObjectDataset(train_img_root, train_scene_json, img_transform=img_preprocess, max_size=100000)
test_dataset = CLEVRObjectDataset(train_img_root, train_scene_json, img_transform=img_preprocess, max_size=1000)
train_dataset, val_dataset = random_split(train_dataset, [90000, 10000])

train_loader = DataLoader(train_dataset.dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset.dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

model = resnet34(pretrained=True)
model.fc = nn.Linear(512, len(train_dataset.dataset.classes))

for param in model.parameters():
    param.requires_grad = False
model.fc.weight.requires_grad = True  # unfreeze last layer weights
model.fc.bias.requires_grad = True  # unfreeze last layer biases


class CLEVRObjectModel(pl.LightningModule):

    def __init__(self):
        super(CLEVRObjectModel, self).__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.criteria = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = self.criteria(y_hat, y)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        val_loss = self.criteria(y_hat, y)
        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        test_loss = self.criteria(y_hat, y)
        return {'test_loss': test_loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=1e-3)  # only optimise non-frozen layers
        return optimizer

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


print('Starting Training...')

clevr_model = CLEVRObjectModel()
trainer = pl.Trainer(max_epochs=10, gpus=1)
trainer.fit(clevr_model)

trainer.test()

img, target = test_dataset[0]
prediction = clevr_model(img.unsqueeze(0).to('cuda:0'))

predicted_class = test_dataset.classes[prediction.argmax(1).cpu()]
print(prediction)
print('The predicted class is', predicted_class)
print('The target is', test_dataset.classes[target])
plt.imshow(img.permute(1, 2, 0))
plt.show()
