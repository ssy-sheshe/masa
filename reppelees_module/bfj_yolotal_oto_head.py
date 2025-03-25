import torch
import torch.nn as nn
import torch.nn.functional as F
from .bfj_yolotal_head import BFJYoloTalHead
from mmdet.models.dense_heads.anchor_head import AnchorHead
from mmdet.registry import MODELS
from mmdet.models.utils import multi_apply
from mmengine.structures import InstanceData
from torch import Tensor
from typing import List, Optional, Tuple, Union
from mmengine.config import ConfigDict
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, OptMultiConfig)

@MODELS.register_module()
class BFJYoloTalOTOHead(AnchorHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 share_head=True,
                 reg_decoded_bbox = True,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                bbox_coder = dict(
                    type='CrystalBoxCoder',
                    weights=[10., 10., 5., 5.]),
                loss_cls=dict(
                    type='YoloVariFocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.75,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='SIoULoss',
                    loss_weight=2.5),
                loss_hook_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                init_cfg=dict(
                    type='Kaiming',
                    layer='Conv2d'),
                topk=13,
                alpha = 1.0,
                beta = 6.0,
                mode='train',
                quality_wise_reg_loss=False,
                train_cfg=None,
                test_cfg=None,):
        self._num_classes = num_classes
        self._in_channels = in_channels
        self._feat_channels = feat_channels
        self._stacked_convs = stacked_convs
        self._conv_cfg = conv_cfg
        self._norm_cfg = norm_cfg
        self._act_cfg = act_cfg
        self._share_head = share_head
        self._reg_decoded_bbox = reg_decoded_bbox
        self._anchor_generator = anchor_generator
        self._bbox_coder = bbox_coder
        self._loss_cls = loss_cls
        self._loss_bbox = loss_bbox
        self._loss_hook_bbox = loss_hook_bbox
        self._init_cfg = init_cfg
        self._topk = topk
        self._alpha = alpha
        self._beta = beta
        self._train_cfg = train_cfg
        self._test_cfg = test_cfg
        self.mode = mode
        self._quality_wise_reg_loss = quality_wise_reg_loss
        super(BFJYoloTalOTOHead, self).__init__(
            num_classes,
            in_channels,
            feat_channels,
            anchor_generator=anchor_generator,
            bbox_coder = bbox_coder,
            reg_decoded_bbox = reg_decoded_bbox,
            loss_cls = loss_cls,
            loss_bbox = loss_bbox,
            train_cfg = train_cfg,
            test_cfg = test_cfg,
            init_cfg=init_cfg)

    def _init_layers(self):
        if self.mode == 'train':
            self.one_to_many_head = BFJYoloTalHead(
                                    num_classes = self._num_classes,
                                    in_channels = self._in_channels,
                                    feat_channels = self._feat_channels,
                                    stacked_convs = self._stacked_convs,
                                    conv_cfg = self._conv_cfg,
                                    norm_cfg = self._norm_cfg,
                                    act_cfg = self._act_cfg,
                                    share_head = self._share_head,
                                    reg_decoded_bbox = self._reg_decoded_bbox,
                                    anchor_generator = self._anchor_generator,
                                    bbox_coder = self._bbox_coder,
                                    loss_cls = self._loss_cls,
                                    loss_bbox = self._loss_bbox,
                                    loss_hook_bbox = self._loss_hook_bbox,
                                    init_cfg = self._init_cfg,
                                    topk = self._topk,
                                    alpha = self._alpha,
                                    beta = self._beta,
                                    quality_wise_reg_loss=self._quality_wise_reg_loss)
            self.one_to_many_head._init_layers()
            self.one_to_many_head.train_cfg = self.train_cfg
            self.one_to_many_head.test_cfg = self.test_cfg
            self.one_to_one_head = BFJYoloTalHead(
                                    num_classes = self._num_classes,
                                    in_channels = self._in_channels,
                                    feat_channels = self._feat_channels,
                                    stacked_convs = self._stacked_convs,
                                    conv_cfg = self._conv_cfg,
                                    norm_cfg = self._norm_cfg,
                                    act_cfg = self._act_cfg,
                                    share_head = self._share_head,
                                    reg_decoded_bbox = self._reg_decoded_bbox,
                                    anchor_generator = self._anchor_generator,
                                    bbox_coder = self._bbox_coder,
                                    loss_cls = self._loss_cls,
                                    loss_bbox = self._loss_bbox,
                                    loss_hook_bbox = self._loss_hook_bbox,
                                    init_cfg = self._init_cfg,
                                    topk = 1,
                                    alpha = self._alpha,
                                    beta = self._beta,
                                    quality_wise_reg_loss=self._quality_wise_reg_loss)
            self.one_to_one_head._init_layers()
            self.one_to_one_head.train_cfg = self.train_cfg
            self.one_to_one_head.test_cfg = self.test_cfg
        else:
            self.one_to_one_head = BFJYoloTalHead(
                                    num_classes = self._num_classes,
                                    in_channels = self._in_channels,
                                    feat_channels = self._feat_channels,
                                    stacked_convs = self._stacked_convs,
                                    conv_cfg = self._conv_cfg,
                                    norm_cfg = self._norm_cfg,
                                    act_cfg = self._act_cfg,
                                    share_head = self._share_head,
                                    reg_decoded_bbox = self._reg_decoded_bbox,
                                    anchor_generator = self._anchor_generator,
                                    bbox_coder = self._bbox_coder,
                                    loss_cls = self._loss_cls,
                                    loss_bbox = self._loss_bbox,
                                    loss_hook_bbox = self._loss_hook_bbox,
                                    init_cfg = self._init_cfg,
                                    topk = 1,
                                    alpha = self._alpha,
                                    beta = self._beta,
                                    quality_wise_reg_loss=self._quality_wise_reg_loss)
            self.one_to_one_head._init_layers()
            self.one_to_one_head.train_cfg = self.train_cfg
            self.one_to_one_head.test_cfg = self.test_cfg
    def init_weights(self):
        if self.mode == 'train':
            self.one_to_many_head.init_weights()
            self.one_to_one_head.init_weights()
        else:
            self.one_to_one_head.init_weights()
    
    def forward(self, feats):
        is_eval = not self.training
        if self.mode == 'train' and not is_eval:
            cls_score = []
            bbox_pred = []
            cls_score0, bbox_pred0 = self.one_to_many_head.forward(feats)
            cls_score.append(cls_score0)
            bbox_pred.append(bbox_pred0)

            cls_score1, bbox_pred1 = self.one_to_one_head.forward(feats)
            cls_score.append(cls_score1)
            bbox_pred.append(bbox_pred1)
        else:
            cls_score, bbox_pred = self.one_to_one_head.forward(feats)
        return cls_score, bbox_pred

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict: A dictionary of loss components.
        """
        loss0 = self.one_to_many_head.loss_by_feat(
            cls_scores[0],
            bbox_preds[0],
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore)
        loss1 = self.one_to_one_head.loss_by_feat(
            cls_scores[1],
            bbox_preds[1],
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore)
        loss_dict = dict()
        for key in loss0.keys():
            loss_dict[key] = (loss0[key] + loss1[key]) * 0.5
        return loss_dict

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        return self.one_to_one_head._predict_by_feat_single(cls_score_list,
                                                            bbox_pred_list,
                                                            score_factor_list, 
                                                            mlvl_priors,
                                                            img_meta,
                                                            cfg,
                                                            rescale,
                                                            with_nms)