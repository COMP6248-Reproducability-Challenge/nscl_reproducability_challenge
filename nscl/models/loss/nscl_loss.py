import torch
import torch.nn as nn

from nscl.datasets.clevr_definition import QuestionTypes

class SceneParsingLoss(nn.Module):
    def __init__(self,  reduction='mean'):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, object_annotations, scenes):
        losses = []
        for object_annotation, scene in zip(object_annotations, scenes):
            for i in range(min(object_annotation.num_objects, len(scene.objects))):
                obj = scene.objects[i]
                actual_concepts = []
                for attr in object_annotation.all_attributes:
                    actual_concepts.append(getattr(obj, attr))

                for c in object_annotation.all_concepts:
                    similarity = object_annotation.similarity(c)[i]
                    expected_similarity = torch.tensor(1., dtype=similarity.dtype, device=similarity.device) if c in actual_concepts else torch.tensor(0., dtype=similarity.dtype, device=similarity.device)
                    losses.append(self.mse_loss(similarity, expected_similarity))
                
        return torch.stack(losses).sum()

class CESceneParsingLoss(nn.Module):
    def __init__(self, definitions, reduction='mean'):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
        self.definitions = definitions

    def get_targets(self, scene, attr):
        targets = [torch.tensor(self.definitions[attr].index(getattr(obj, attr))) for obj in scene.objects]
        return torch.stack(targets)

    def get_predictions(self, object_annotation, attr):
        predictions = [object_annotation.similarity(concept) for concept in self.definitions[attr]]
        return torch.stack(predictions).t()

    def compute_loss(self, object_annotation, scene):
        losses = []
        for attr in self.definitions.keys():
            targets = self.get_targets(scene, attr)
            predictions = self.get_predictions(object_annotation, attr)
            losses.append(self.ce_loss(predictions, targets.to(predictions.device)))
        return torch.stack(losses).sum()

    def forward(self, object_annotations, scenes):
        losses = []
        for a, s in zip(object_annotations, scenes):
            if len(s.objects) != a.num_objects: continue
            losses.append(self.compute_loss(a, s))

        return torch.stack(losses).sum()

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
            
            losses.append(loss_function(predict, actual))

        return torch.stack(losses).sum()
