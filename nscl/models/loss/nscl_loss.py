import torch.nn as nn
import torch.nn.modules.loss
from nscl.datasets.clevr_definition import QuestionTypes

class SceneParsingLoss(nn.Module):
    def __init__(self,  reduction='mean'):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, object_annotation, scene):
        losses = []
        for i in range(scene.objects):
            obj = scene.objects[i]
            for attr in object_annotation.all_attributes:
                actual_concept = getattr(obj, attr)
                similarity = object_annotation.similarity(actual_concept)[i]
                losses.append(1. - similarity)

        return torch.cat(losses).sum()

class QALoss(nn.Module):
    def __init__(self,  reduction='mean'):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.bce_loss = nn.BCELoss(reduction=reduction)
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, questions, predicts):
        losses = []
        for q, predict in zip(questions, predicts):
            actual = q.answer_tensor.to(predict.device)
            if q.question_type == QuestionTypes.BOOLEAN:
                loss_function = self.bce_loss
                predict = torch.stack([predict, 1. - predict]).to(predict.device)
            elif q.question_type == QuestionTypes.COUNT:
                loss_function = self.mse_loss
            elif q.answer_tensor.shape[0] == 2:
                loss_function = self.bce_loss
            else:
                loss_function = self.ce_loss
                predict = predict.unsqueeze(0)
            
            losses.append(loss_function(predict, actual).unsqueeze(0))

        return torch.cat(losses).sum()
