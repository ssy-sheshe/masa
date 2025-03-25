# Copyright (c) OpenMMLab. All rights reserved.
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from torch import Tensor
from mmengine.structures import InstanceData
from mmcv.ops import batched_nms
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, OptMultiConfig,reduce_mean)
from mmcv.cnn import ConvModule
from mmengine.model import bias_init_with_prob, normal_init
from mmdet.models.utils import multi_apply,filter_scores_and_topk
from mmdet.registry import MODELS
from mmdet.models.dense_heads.anchor_head import AnchorHead
from .yolo_varifocal_loss import YoloVariFocalLoss
from scipy.optimize import linear_sum_assignment
import numpy as np
from mmengine.config import ConfigDict
from mmdet.structures.bbox import (cat_boxes, get_box_tensor, get_box_wh,
                                   scale_boxes)

@MODELS.register_module()
class BFJYoloTalHead(AnchorHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 share_head=False,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 loss_hook_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 topk=13,
                 alpha = 1.0,
                 beta = 6.0,
                 num_classes=2,
                 quality_wise_reg_loss=False,
                 **kwargs):
        
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.share_head = share_head
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.quality_wise_reg_loss = quality_wise_reg_loss
        if self.quality_wise_reg_loss:
            print('quality_wise_reg_loss is set to True !!!')
        super(BFJYoloTalHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)
        print('TaskAlignedAssigner topk: {} alpha: {} beta {}'.format(self.topk, self.alpha, self.beta))
        self.tal_assigner = TaskAlignedAssigner(box_coder=self.bbox_coder, 
                                                num_classes=self.num_classes, 
                                                topk=self.topk,
                                                alpha=self.alpha,
                                                beta=self.beta)
        # self.loss_hook_bbox = build_loss(loss_hook_bbox)
        self.loss_hook_bbox = MODELS.build(loss_hook_bbox)

    def _init_layers(self):
        """Initialize layers of the head."""
        if not self.share_head:
            self.cls_convs = nn.ModuleList()
            self.reg_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                self.cls_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                self.reg_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
        else:
            self.shared_head_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                self.shared_head_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
        
        self.retina_cls = nn.Conv2d(self.feat_channels,self.num_base_priors * self.cls_out_channels,3,padding=1)
        self.retina_reg = nn.Conv2d(self.feat_channels, self.num_base_priors * 8, 3, padding=1)
    
    def init_weights(self):
        """Initialize weights of the head."""

        if not self.share_head:
            for m in self.cls_convs:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01, bias=0)
            for m in self.reg_convs:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01, bias=0)
        else:
            for m in self.shared_head_convs:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01, bias=0)

        normal_init(self.retina_cls, std=0.01, bias=bias_init_with_prob(0.01))
        normal_init(self.retina_reg, std=0.01, bias=0)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        if not self.share_head:
            cls_feat = x
            reg_feat = x
            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(reg_feat)

        else:
            for conv in self.shared_head_convs:
                x = conv(x)
            cls_feat = x
            reg_feat = x
                
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)

        return cls_score, bbox_pred
    
    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
        """
        return multi_apply(self.forward_single, feats)

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
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, batch_img_metas, device=device)
        labels, regression_targets, bbox_hook_targets, scores_targets, target_mask = self.tal_assigner(anchor_list, batch_gt_instances, cls_scores, bbox_preds)
        N = len(labels)

        cls_scores, bbox_preds = concat_box_prediction_layers(cls_scores, bbox_preds)

        labels = torch.cat(labels, dim=1).squeeze(0)
        regression_targets = torch.cat(regression_targets, dim=1).squeeze(0)
        bbox_hook_targets = torch.cat(bbox_hook_targets, dim=1).squeeze(0)
        scores_targets = torch.cat(scores_targets, dim=1).squeeze(0)
        target_mask = torch.cat(target_mask, dim=1).squeeze(0)
        # print(labels.size(), scores_targets.size(), target_mask.size())

        pos_inds = (labels >= 0) & (labels != self.num_classes)
        num_pos_anchors = pos_inds.sum().item()
        if self.reg_decoded_bbox:
            anchors_over_all_imgs = torch.cat([torch.cat(anchors, dim=0) for anchors in anchor_list], dim=0)
            bbox_pred = self.bbox_coder.decode(anchors_over_all_imgs, bbox_preds[..., :4])
            bbox_hook_pred = self.bbox_coder.decode(anchors_over_all_imgs, bbox_preds[..., 4:])
        bbox_weights = torch.zeros_like(regression_targets)
        bbox_weights[pos_inds, :] = 1.
        if self.quality_wise_reg_loss:
            box_quality,_ = torch.max(cls_scores.detach().sigmoid()** 2, dim=-1)
            box_quality = box_quality.unsqueeze(-1).repeat(1,4)
            bbox_weights = bbox_weights * box_quality

        losses_bbox = self.loss_bbox(
            bbox_pred,
            regression_targets,
            bbox_weights,
            avg_factor=(num_pos_anchors))

        losses_hook_bbox = self.loss_hook_bbox(
            bbox_hook_pred, 
            bbox_hook_targets, 
            bbox_weights, 
            avg_factor=(num_pos_anchors))

        if isinstance(self.loss_cls, YoloVariFocalLoss):
            losses_cls = self.loss_cls(cls_scores, labels, scores_targets, weight=target_mask, avg_factor=(num_pos_anchors + N))
        else:
            losses_cls = self.loss_cls(cls_scores, labels, weight=target_mask, avg_factor=(num_pos_anchors + N))

        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_hook_bbox=losses_hook_bbox)

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmengine.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        if score_factor_list[0] is None:
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_hook_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              score_factor_list, mlvl_priors)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            # dim = self.bbox_coder.encode_size
            # bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, dim)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 8)
            hook_pred = bbox_pred[..., 4:]
            bbox_pred = bbox_pred[..., :4]
            if with_score_factors:
                score_factor = score_factor.permute(1, 2,
                                                    0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)

            # the `custom_cls_channels` parameter is derived from
            # CrossEntropyCustomLoss and FocalCustomLoss, and is currently used
            # in v3det.
            if getattr(self.loss_cls, 'custom_cls_channels', False):
                scores = self.loss_cls.get_activation(cls_score)
            elif self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            score_thr = cfg.get('score_thr', 0)

            results = filter_scores_and_topk(
                scores, score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors, hook_pred=hook_pred))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']
            hook_pred = filtered_results['hook_pred']

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_hook_preds.append(hook_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        hook_pred = torch.cat(mlvl_hook_preds)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)
        hooks = self.bbox_coder.decode(priors, hook_pred, max_shape=img_shape)
        scores = torch.cat(mlvl_scores)
        labels = torch.cat(mlvl_labels)

        #use body's hook as head bbox
        first_class_indx = labels==0
        second_class_indx = labels==1
        hook_boxes = hooks[first_class_indx,:]
        hook_scores = scores[first_class_indx]

        hook_labels = labels[first_class_indx] * 0 + 1

        labels[first_class_indx] = 0
        labels[second_class_indx] = 2

        # print(bboxes.size(), hooks.size(), labels.size(), hooks.size())

        bboxes = torch.cat([bboxes, hook_boxes], dim=0)
        scores = torch.cat([scores, hook_scores], dim=0)
        labels = torch.cat([labels, hook_labels], dim=0)

        # print(bboxes.size(), hooks.size(), labels.size(), hooks.size())

        results = InstanceData()
        results.bboxes = bboxes
        # results.hooks = hooks
        results.scores = scores
        results.labels = labels
        if with_score_factors:
            results.score_factors = torch.cat(mlvl_score_factors)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)

    def _bbox_post_process(self,
                           results: InstanceData,
                           cfg: ConfigDict,
                           rescale: bool = False,
                           with_nms: bool = True,
                           img_meta: Optional[dict] = None) -> InstanceData:
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            results (:obj:`InstaceData`): Detection instance results,
                each item has shape (num_bboxes, ).
            cfg (ConfigDict): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default to False.
            with_nms (bool): If True, do nms before return boxes.
                Default to True.
            img_meta (dict, optional): Image meta info. Defaults to None.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)
            # results.hooks = scale_boxes(results.hooks, scale_factor)

        if hasattr(results, 'score_factors'):
            # TODO: Add sqrt operation in order to be consistent with
            #  the paper.
            score_factors = results.pop('score_factors')
            results.scores = results.scores * score_factors

        # filter small size bboxes
        if cfg.get('min_bbox_size', -1) >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        # TODO: deal with `with_nms` and `nms_cfg=None` in test_cfg
        if with_nms and results.bboxes.numel() > 0:
            bboxes = get_box_tensor(results.bboxes)
            det_bboxes, keep_idxs = batched_nms(bboxes, results.scores,
                                                results.labels, cfg.nms)
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[:cfg.max_per_img]

        return results


def generate_hook_gt(batch_gt_instances: InstanceList):
    device = batch_gt_instances[0].bboxes.device
    gt_bboxes, gt_hooks, gt_labels =[], [], []
    for batch_gt_ins in batch_gt_instances:
        tmp_bboxes = batch_gt_ins.bboxes
        tmp_labels = batch_gt_ins.labels
        # print(tmp_labels.size(), tmp_bboxes.size())
        if tmp_labels.size(0) ==0:
            gt_bboxes.append(torch.tensor([]).to(device)) 
            gt_hooks.append(torch.tensor([]).to(device))
            gt_labels.append(torch.tensor([]).to(device))
            continue

        device = tmp_bboxes.device
        class_wise_data = dict()
        class_wise_data[0] = {'label':[],'bbox':[],'bbox_np':[],'hook':[]} #body
        class_wise_data[1] = {'label':[],'bbox':[],'bbox_np':[],'hook':[]} #head
        class_wise_data[2] = {'label':[],'bbox':[],'bbox_np':[],'hook':[]} #face
        for bbox, label in zip(tmp_bboxes, tmp_labels):
            key = int(label.detach().cpu().numpy())
            class_wise_data[key]['label'].append(label)
            class_wise_data[key]['bbox'].append(bbox.unsqueeze(0))
            class_wise_data[key]['bbox_np'].append(bbox.detach().cpu().numpy())
            class_wise_data[key]['hook'].append(bbox.unsqueeze(0).clone())

        if len(class_wise_data[1]['bbox_np']) > 0:
            body_head_associate_result = associate_body_head_by_hungarian_algorithm(class_wise_data[0]['bbox_np'], class_wise_data[1]['bbox_np'])
            for body_id, body_box in enumerate(class_wise_data[0]['bbox']):
                if body_id in body_head_associate_result[:, 0]:
                    idx = np.where(body_head_associate_result[:, 0] == body_id)[0][0]
                    head_id = body_head_associate_result[idx, 1]
                    if head_id != -1:
                        class_wise_data[0]['hook'][body_id] = class_wise_data[1]['bbox'][head_id].clone()

            if len(class_wise_data[2]['bbox_np']) > 0:
                head_face_associate_result = associate_body_head_by_hungarian_algorithm(class_wise_data[1]['bbox_np'], class_wise_data[2]['bbox_np'])
                for head_id, head_box in enumerate(class_wise_data[1]['bbox']):
                    if head_id in head_face_associate_result[:, 0]:
                        idx = np.where(head_face_associate_result[:, 0] == head_id)[0][0]
                        face_id = head_face_associate_result[idx, 1]
                        if face_id != -1:
                            class_wise_data[2]['hook'][face_id] = class_wise_data[1]['bbox'][head_id].clone()
            
        labels = None
        bboxes = None
        hooks = None
        for idx, label in  enumerate(class_wise_data.keys()):
            if (label == 0 or label == 2)  and len(class_wise_data[label]['bbox']) > 0:
                assert len(class_wise_data[label]['bbox']) == len(class_wise_data[label]['hook'])
                new_label = 1 if label == 2 else 0
                if labels is None:
                    bboxes = torch.cat(class_wise_data[label]['bbox'],dim=0)
                    hooks = torch.cat(class_wise_data[label]['hook'],dim=0)
                    labels = torch.ones(len(class_wise_data[label]['bbox'])).to(device) * new_label
                else:
                    bboxes = torch.cat([bboxes, torch.cat(class_wise_data[label]['bbox'],dim=0)],dim=0)
                    hooks = torch.cat([hooks, torch.cat(class_wise_data[label]['hook'],dim=0)],dim=0)
                    tmp_labels = torch.ones(len(class_wise_data[label]['bbox'])).to(device) * new_label
                    labels = torch.cat([labels, tmp_labels],dim=0)
        if labels is None:
            gt_bboxes.append(torch.tensor([]).to(device)) 
            gt_hooks.append(torch.tensor([]).to(device))
            gt_labels.append(torch.tensor([]).to(device))
        else:
            gt_bboxes.append(bboxes)
            gt_hooks.append(hooks)
            gt_labels.append(labels)
    # print(len(gt_bboxes), len(gt_hooks), len(gt_labels))
    return gt_bboxes, gt_hooks, gt_labels

MAX_VAL = 8e6

def one_side_iou(xyxy_box1, xyxy_box2):
    # 1. to corner box
    box1 = np.asarray(xyxy_box1)
    box2 = np.asarray(xyxy_box2)

    # 2. min and max
    box1 = np.asarray(box1, dtype=float)
    box2 = np.asarray(box2, dtype=float)
    x1 = max(box1[0], box2[0])
    x2 = min(box1[2], box2[2])
    y1 = max(box1[1], box2[1])
    y2 = min(box1[3], box2[3])

    intersection = max(x2 - x1, 0) * max(y2 - y1, 0)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])

    iou = intersection / a1  # intersection over box 1
    # print "inter={}, union={}".format(str(intersection), str(union))
    return iou

def cal_body_face_distance_matrix(body_boxes, face_boxes):
    body_boxes_nums = len(body_boxes)
    face_boxes_nums = len(face_boxes)
    body_face_distance_matrix = np.zeros((body_boxes_nums, face_boxes_nums))

    for body_idx in range(body_boxes_nums):
        body_box = body_boxes[body_idx]
        for face_idx in range(face_boxes_nums):
            face_box = face_boxes[face_idx]
            face_iou_in_body = one_side_iou(face_box, body_box)
            if face_iou_in_body > 0.2:
                body_face_distance_matrix[body_idx, face_idx] = 1 / face_iou_in_body
            else:
                body_face_distance_matrix[body_idx, face_idx] = MAX_VAL

    return body_face_distance_matrix

def associate_body_head_by_hungarian_algorithm(body_result, head_result):
    associate_result = []
    body_boxes = body_result
    head_boxes = head_result
    
    body_face_distance_matrix = cal_body_face_distance_matrix(body_boxes, head_boxes)
    body_row_idxs, face_col_idxs = linear_sum_assignment(body_face_distance_matrix)
    match_id_matrix = np.vstack((body_row_idxs, face_col_idxs)).T

    for idx, (body_idx, head_idx) in enumerate(zip(body_row_idxs, face_col_idxs)):
        if body_face_distance_matrix[body_idx, head_idx] == MAX_VAL:
            match_id_matrix[idx][1] = -1
    return match_id_matrix

def dist_calculator(gt_bboxes, anchor_bboxes):
    """compute center distance between all bbox and gt

    Args:
        gt_bboxes (Tensor): shape(bs*n_max_boxes, 4)
        anchor_bboxes (Tensor): shape(num_total_anchors, 4)
    Return:
        distances (Tensor): shape(bs*n_max_boxes, num_total_anchors)
        ac_points (Tensor): shape(num_total_anchors, 2)
    """
    gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
    gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
    gt_points = torch.stack([gt_cx, gt_cy], dim=1)
    ac_cx = (anchor_bboxes[:, 0] + anchor_bboxes[:, 2]) / 2.0
    ac_cy = (anchor_bboxes[:, 1] + anchor_bboxes[:, 3]) / 2.0
    ac_points = torch.stack([ac_cx, ac_cy], dim=1)

    distances = (gt_points[:, None, :] - ac_points[None, :, :]).pow(2).sum(-1).sqrt()

    return distances, ac_points

def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
    """select the positive anchors's center in gt

    Args:
        xy_centers (Tensor): shape(bs*n_max_boxes, num_total_anchors, 4)
        gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
    Return:
        (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    """
    n_anchors = xy_centers.size(0)
    bs, n_max_boxes, _ = gt_bboxes.size()
    _gt_bboxes = gt_bboxes.reshape([-1, 4])
    xy_centers = xy_centers.unsqueeze(0).repeat(bs * n_max_boxes, 1, 1)
    b_lt = _gt_bboxes[:, 0:2].unsqueeze(1).repeat(1, n_anchors, 1)
    b_rb = _gt_bboxes[:, 2:4].unsqueeze(1).repeat(1, n_anchors, 1)
    b_lt = xy_centers - b_lt
    b_rb = b_rb - xy_centers
    bbox_deltas = torch.cat([b_lt, b_rb], dim=-1)
    bbox_deltas = bbox_deltas.reshape([bs, n_max_boxes, n_anchors, -1])
    bbox_deltas = bbox_deltas.min(axis=-1)[0] > eps
    bbox_deltas = bbox_deltas.to(gt_bboxes.dtype)
    return bbox_deltas

def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """if an anchor box is assigned to multiple gts,
        the one with the highest iou will be selected.

    Args:
        mask_pos (Tensor): shape(bs, n_max_boxes, num_total_anchors)
        overlaps (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    Return:
        target_gt_idx (Tensor): shape(bs, num_total_anchors)
        fg_mask (Tensor): shape(bs, num_total_anchors)
        mask_pos (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    """
    fg_mask = mask_pos.sum(axis=-2)
    if fg_mask.max() > 1:
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])
        max_overlaps_idx = overlaps.argmax(axis=1)
        is_max_overlaps = F.one_hot(max_overlaps_idx, n_max_boxes)
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
        fg_mask = mask_pos.sum(axis=-2)
    target_gt_idx = mask_pos.argmax(axis=-2)
    return target_gt_idx, fg_mask , mask_pos

def iou_calculator(box1, box2, eps=1e-9):
    """Calculate iou for batch

    Args:
        box1 (Tensor): shape(bs, n_max_boxes, 1, 4)
        box2 (Tensor): shape(bs, 1, num_total_anchors, 4)
    Return:
        (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    """
    box1 = box1.unsqueeze(2)  # [N, M1, 4] -> [N, M1, 1, 4]
    box2 = box2.unsqueeze(1)  # [N, M2, 4] -> [N, 1, M2, 4]
    px1y1, px2y2 = box1[:, :, :, 0:2], box1[:, :, :, 2:4]
    gx1y1, gx2y2 = box2[:, :, :, 0:2], box2[:, :, :, 2:4]
    x1y1 = torch.maximum(px1y1, gx1y1)
    x2y2 = torch.minimum(px2y2, gx2y2)
    overlap = (x2y2 - x1y1).clip(0).prod(-1)
    area1 = (px2y2 - px1y1).clip(0).prod(-1)
    area2 = (gx2y2 - gx1y1).clip(0).prod(-1)
    union = area1 + area2 - overlap + eps

    return overlap / union

def permute_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (A x K), H, W) to (N, (HxWxA), K)
    Used in the label assignment of OTA
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor

def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer

def concat_box_prediction_layers(box_cls, box_regression):
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(
            box_cls, box_regression
    ):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 8
        C = AxC // A
        box_cls_per_level = permute_and_flatten(
            box_cls_per_level, N, A, C, H, W
        )
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(
            box_regression_per_level, N, A, 8, H, W
        )
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = torch.cat(box_cls_flattened, dim=1).reshape(-1, C)
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 8)
    return box_cls, box_regression

def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)


def centerness_calculator(gt_bboxes, pred_bboxes, eps):

    gt_x0 = gt_bboxes[..., 0] 
    gt_y0 = gt_bboxes[..., 1] 
    gt_x1 = gt_bboxes[..., 2]
    gt_y1 = gt_bboxes[..., 3]

    pd_cx = (pred_bboxes[..., 0] + pred_bboxes[..., 2]) / 2.0
    pd_cy = (pred_bboxes[..., 1] + pred_bboxes[..., 3]) / 2.0
    l = torch.abs(pd_cx[:, None, :] - gt_x0[:, :, None]) + eps
    r = torch.abs(pd_cx[:, None, :] - gt_x1[:, :, None]) + eps
    t = torch.abs(pd_cy[:, None, :] - gt_y0[:, :, None]) + eps
    b = torch.abs(pd_cy[:, None, :] - gt_y1[:, :, None]) + eps
    centerness = torch.sqrt(l.min(r)/l.max(r) *  t.min(b)/t.max(b))

    return centerness

def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    FP16 Contributed by https://github.com/open-mmlab/mmdetection/pull/4889
    Note:
        Assume bboxes1 is M x 4, bboxes2 is N x 4, when mode is 'iou',
        there are some new generated variable when calculating IOU
        using bbox_overlaps function:

        1) is_aligned is False
            area1: M x 1
            area2: N x 1
            lt: M x N x 2
            rb: M x N x 2
            wh: M x N x 2
            overlap: M x N x 1
            union: M x N x 1
            ious: M x N x 1

            Total memory:
                S = (9 x N x M + N + M) * 4 Byte,

            When using FP16, we can reduce:
                R = (9 x N x M + N + M) * 4 / 2 Byte
                R large than (N + M) * 4 * 2 is always true when N and M >= 1.
                Obviously, N + M <= N * M < 3 * N * M, when N >=2 and M >=2,
                           N + 1 < 3 * N, when N or M is 1.

            Given M = 40 (ground truth), N = 400000 (three anchor boxes
            in per grid, FPN, R-CNNs),
                R = 275 MB (one times)

            A special case (dense detection), M = 512 (ground truth),
                R = 3516 MB = 3.43 GB

            When the batch size is B, reduce:
                B x R

            Therefore, CUDA memory runs out frequently.

            Experiments on GeForce RTX 2080Ti (11019 MiB):

            |   dtype   |   M   |   N   |   Use    |   Real   |   Ideal   |
            |:----:|:----:|:----:|:----:|:----:|:----:|
            |   FP32   |   512 | 400000 | 8020 MiB |   --   |   --   |
            |   FP16   |   512 | 400000 |   4504 MiB | 3516 MiB | 3516 MiB |
            |   FP32   |   40 | 400000 |   1540 MiB |   --   |   --   |
            |   FP16   |   40 | 400000 |   1264 MiB |   276MiB   | 275 MiB |

        2) is_aligned is True
            area1: N x 1
            area2: N x 1
            lt: N x 2
            rb: N x 2
            wh: N x 2
            overlap: N x 1
            union: N x 1
            ious: N x 1

            Total memory:
                S = 11 x N * 4 Byte

            When using FP16, we can reduce:
                R = 11 x N * 4 / 2 Byte

        So do the 'giou' (large than 'iou').

        Time-wise, FP16 is generally faster than FP32.

        When gpu_assign_thr is not -1, it takes more time on cpu
        but not reduce memory.
        There, we can reduce half the memory and keep the speed.

    If ``is_aligned`` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned`` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )

    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious

class TaskAlignedAssigner(nn.Module):
    def __init__(self,
                 box_coder,
                 topk=13,
                 num_classes=80,
                 alpha=1.0,
                 beta=6.0,
                 eps=1e-9):
        super(TaskAlignedAssigner, self).__init__()
        self.init_topk = topk
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.box_coder = box_coder

    @torch.no_grad()
    def forward(self,
                anchors,
                batch_gt_instances,
                box_cls,
                box_regression):
        """This code referenced to
           https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)
        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
        """

        box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_cls = torch.cat(box_cls, dim=1)
        box_regression = [permute_to_N_HWA_K(x, 8) for x in box_regression]
        box_regression = torch.cat(box_regression, dim=1)
        all_target_labels = []
        all_target_bboxes = []
        all_target_hooks = []
        all_target_scores = []
        all_target_mask = []
        targets_boxes_list, targets_hooks_list, targets_labels_list = generate_hook_gt(batch_gt_instances)
        for anchors_per_image, gt_bboxes, gt_hooks, gt_labels, pd_scores, \
            box_regression_per_image in zip(anchors, targets_boxes_list, targets_hooks_list, targets_labels_list, box_cls, box_regression):

            device = gt_bboxes.device
        
            anchors_over_all = torch.cat([anchor_per_image for anchor_per_image in anchors_per_image], dim=0)
            anc_points = (anchors_over_all[:, :2] + anchors_over_all[:, 2:]) / 2
            pd_bboxes = self.box_coder.decode(anchors_over_all, box_regression_per_image[..., :4]).unsqueeze(0)
            pd_hooks = self.box_coder.decode(anchors_over_all, box_regression_per_image[..., 4:]).unsqueeze(0)
            pd_scores = F.sigmoid(pd_scores).unsqueeze(0)
            gt_bboxes = gt_bboxes.unsqueeze(0)
            gt_hooks = gt_hooks.unsqueeze(0)
            gt_labels = gt_labels.unsqueeze(0).unsqueeze(-1)
            mask_gt = torch.ones(gt_labels.size()).to(device)
            self.bs = pd_scores.size(0)
            self.n_max_boxes = gt_bboxes.size(1)
            

            if self.n_max_boxes == 0:
                target_labels = torch.full_like(pd_scores[..., 0], self.bg_idx).to(device)
                target_bboxes = torch.zeros_like(pd_bboxes).to(device)#.squeeze(0)
                target_hooks = torch.zeros_like(pd_bboxes).to(device)
                target_scores = torch.zeros_like(pd_scores).to(device)
                target_mask = torch.zeros_like(pd_scores[..., 0]).to(device)

            else:
                target_labels, target_bboxes, target_hooks, target_scores, target_gt_idx = self.get_targets(pd_scores, 
                                            pd_bboxes, pd_hooks, gt_labels, gt_bboxes, gt_hooks, anc_points, mask_gt)
                target_mask = torch.ones_like(pd_scores[..., 0]).to(device)
                
            all_target_labels.append(target_labels.long())
            all_target_bboxes.append(target_bboxes)
            all_target_hooks.append(target_hooks)
            all_target_scores.append(target_scores)
            all_target_mask.append(target_mask)

        return all_target_labels, all_target_bboxes, all_target_hooks, all_target_scores, all_target_mask

    def get_pos_mask(self,
                     pd_scores,
                     pd_bboxes,
                     pd_hooks,
                     gt_labels,
                     gt_bboxes,
                     gt_hooks,
                     anc_points,
                     mask_gt):

        # get anchor_align metric
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, pd_hooks, gt_labels, gt_bboxes, gt_hooks)
        # get in_gts mask
        # process gt boxes one by one, prevent from out of memory
        num_gt = gt_bboxes.size(1)
        mask = []
        for i in range(0, num_gt):
            mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes[:,i,:].unsqueeze(1))
            mask.append(mask_in_gts)
        mask = torch.cat(mask,dim=1)
        # mask = torch.ones_like(align_metric)
        # get topk_metric mask
        mask = self.select_topk_candidates(align_metric,  mask, topk_mask=mask_gt.repeat([1, 1, self.topk]).bool())
        mask = mask * mask_gt

        return mask, align_metric, overlaps

    def get_box_metrics(self,
                        pd_scores,
                        pd_bboxes,
                        pd_hooks,
                        gt_labels,
                        gt_bboxes,
                        gt_hooks):

        pd_scores = pd_scores.permute(0, 2, 1)
        gt_labels = gt_labels.to(torch.long)
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1)
        pd_scores = pd_scores[ind[0], ind[1]]

        # process gt boxes one by one, prevent from out of memory
        num_gt = gt_bboxes.size(1)
        overlaps = []
        for i in range(0, num_gt):
            overlap0 = iou_calculator(gt_bboxes[:,i,:].unsqueeze(1), pd_bboxes)
            overlap1 = iou_calculator(gt_hooks[:,i,:].unsqueeze(1), pd_hooks)
            overlap = (overlap0  +  overlap1) * 0.5
            overlaps.append(overlap0)
        overlaps = torch.cat(overlaps, dim=1)

        # num_gt = gt_bboxes.size(1)
        # overlaps = []
        # for i in range(0, num_gt):
        #     overlap = bbox_overlaps(gt_bboxes[:,i,:].unsqueeze(1), pd_bboxes, mode='giou', is_aligned=False, eps=self.eps)
        #     overlaps.append(overlap)
        # overlaps = torch.cat(overlaps, dim=1)
        # overlaps = overlaps.sigmoid()

        # overlaps = bbox_overlaps(gt_bboxes, pd_bboxes, mode='giou', is_aligned=False, eps=self.eps)
        # overlaps = overlaps.sigmoid()
        #centerness = centerness_calculator(gt_bboxes, pd_bboxes, self.eps)
        
        align_metric = pd_scores.pow(self.alpha) * overlaps.pow(self.beta) #* centerness

        return align_metric, overlaps
    
    def select_topk_candidates(self,
                               metrics,
                               mask,
                               largest=True,
                               topk_mask=None):

        num_anchors = metrics.shape[-1]
        metrics = metrics * mask
        metrics, topk_idxs = torch.topk(metrics, self.topk, axis=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (metrics.max(axis=-1, keepdim=True) > self.eps).tile([1, 1, self.topk])
        topk_mask = topk_mask.to(topk_idxs.device)
        topk_idxs = torch.where(topk_mask, topk_idxs, torch.zeros_like(topk_idxs))
        for i in range(0, topk_idxs.size(1)):
            topk_idxs_tmp = topk_idxs[:,i,:].unsqueeze(1)
            is_in_topk = F.one_hot(topk_idxs_tmp, num_anchors).sum(axis=-2)
            mask[:,i,:] *= torch.where(is_in_topk > 1, torch.zeros_like(is_in_topk), is_in_topk).squeeze(1)
        return mask

    def get_targets(self,
                    pd_scores,
                    pd_bboxes,
                    pd_hooks,
                    gt_labels,
                    gt_bboxes,
                    gt_hooks,
                    anc_points,
                    mask_gt):
        
        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, pd_hooks, gt_labels, gt_bboxes,gt_hooks, anc_points, mask_gt)
        if self.topk > 1:
            target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)
        else:
            fg_mask = mask_pos.sum(axis=-2)
            target_gt_idx = mask_pos.argmax(axis=-2)

        # assigned target labels
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[...,None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
        gt_labels = gt_labels.long().flatten()[target_gt_idx]

        # assigned target boxes
        gt_bboxes = gt_bboxes.reshape([-1, 4])[target_gt_idx]
        gt_hooks = gt_hooks.reshape([-1, 4])[target_gt_idx]

        # assigned target scores
        gt_labels[gt_labels<0] = 0
        target_scores = F.one_hot(gt_labels, self.num_classes)
        fg_scores_mask  = fg_mask[:, :, None].repeat(1, 1, self.num_classes)
        target_scores = torch.where(fg_scores_mask > 0, target_scores,
                                        torch.full_like(target_scores, 0))
        gt_labels = torch.where(fg_mask > 0, gt_labels, torch.full_like(gt_labels, self.num_classes))
        # normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.max(axis=-1, keepdim=True)[0]
        pos_overlaps = (overlaps * mask_pos).max(axis=-1, keepdim=True)[0]
        align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).max(-2)[0].unsqueeze(-1)
        target_scores = target_scores * align_metric

        return gt_labels, gt_bboxes, gt_hooks, target_scores, target_gt_idx
