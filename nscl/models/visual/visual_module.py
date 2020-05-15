import torch
from torch import nn
from torchvision.models import resnet34
from torchvision.ops import RoIAlign

from nscl.models.visual.functional import normalize

__all__ = ['VisualModule']


class VisualModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.image_size = 256
        self.feature_dim = 256
        self.output_dim = 256
        self.pool_size = 7
        self.downsample_rate = 16

        self.roi_align = RoIAlign(self.pool_size, 1.0 / self.downsample_rate, -1)
        self.context_feature_extract = nn.Conv2d(self.feature_dim, self.feature_dim, 1)
        self.object_feature_fuse = nn.Conv2d(self.feature_dim * 2, self.output_dim, 1)
        self.object_feature_fc = nn.Sequential(nn.ReLU(True),
                                               nn.Linear(self.output_dim * self.pool_size ** 2, self.output_dim))

        self.resnet = resnet34(pretrained=True)
        self.resnet_feature_extractor = nn.Sequential(*list(self.resnet.children())[:-3])

        # disable training resnet for now
        for param in self.resnet_feature_extractor.parameters():
            param.requires_grad = False

        # Confirm if we need this?
        self.resnet_feature_extractor.eval()

    def forward(self, images, scenes):
        outputs = []
        image_features = self.resnet_feature_extractor(images)
        context_features = self.context_feature_extract(image_features)
        batch_size = images.size(0)

        for idx in range(batch_size):
            boxes = scenes[idx].boxes
            batch_idx = torch.zeros(boxes.size(0), 1) + idx
            boxes = torch.cat([batch_idx, boxes], dim=-1)

            with torch.no_grad():
                image_h, image_w = image_features.size(2) * self.downsample_rate, image_features.size(
                    3) * self.downsample_rate
                image_box = torch.cat([
                    torch.zeros(boxes.size(0), 1),
                    torch.zeros(boxes.size(0), 1),
                    torch.zeros(boxes.size(0), 1),
                    image_w + torch.zeros(boxes.size(0), 1),
                    image_h + torch.zeros(boxes.size(0), 1)
                ], dim=-1)

            box_feature = self.roi_align(image_features, boxes)
            this_context_features = self.roi_align(context_features, image_box)
            this_object_features = self.object_feature_fuse(torch.cat([box_feature, this_context_features], dim=1))
            outputs.append(normalize(self.object_feature_fc(this_object_features.view(box_feature.size(0), -1))))

        return outputs
