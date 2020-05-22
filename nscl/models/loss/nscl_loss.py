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
    def __init__(self,  reduction='mean'):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, object_annotations, scenes):
        losses = []
        for object_annotation, scene in zip(object_annotations, scenes):
            for attr, concetps in object_annotation.definitions.items():
                all_obj_similarities = []
                all_obj_targets = []
                for i in range(min(object_annotation.num_objects, len(scene.objects))):
                    similarities = torch.stack([object_annotation.similarity(c)[i] for c in concetps])
                    target = torch.tensor(concetps.index(getattr(scene.objects[i], attr)), dtype=torch.long, device=similarities.device)
                    all_obj_similarities.append(similarities)
                    all_obj_targets.append(target)
                
                all_obj_similarities = torch.stack(all_obj_similarities)
                all_obj_targets = torch.stack(all_obj_targets)
                losses.append(self.ce_loss(all_obj_similarities, all_obj_targets))
                
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
