# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmdet.models.losses.utils import weight_reduce_loss

def yolo_vari_focal_loss(pred,
                       target,
                       target_scores,
                       weight=None,
                       gamma=2.0,
                       alpha=0.75,
                       reduction='mean',
                       avg_factor=None):

    pred_score = torch.sigmoid(pred)
    num_classes = pred_score.size(1)
    label = F.one_hot(target, num_classes + 1)[..., :-1]
    weight = alpha * pred_score.pow(gamma) * (1 - label) + target_scores * label
    with torch.cuda.amp.autocast(enabled=False):
        loss = (F.binary_cross_entropy(pred_score.float(), target_scores.float(), reduction='none') * weight)
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

@MODELS.register_module()
class YoloVariFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.75,
                 reduction='mean',
                 loss_weight=1.0,
                 activated=False):
        super(YoloVariFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated

    def forward(self,
                pred,
                target,
                target_scores,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * yolo_vari_focal_loss(
                pred,
                target,
                target_scores,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)

        else:
            raise NotImplementedError
        return loss_cls
