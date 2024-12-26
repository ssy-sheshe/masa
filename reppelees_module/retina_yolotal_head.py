# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import bias_init_with_prob, normal_init
# from mmcv.runner import force_fp32
from mmdet.models.utils import multi_apply
from mmdet.registry import MODELS
# from .anchor_head import AnchorHead
from mmdet.models.dense_heads.anchor_head import AnchorHead
from .yolo_varifocal_loss import YoloVariFocalLoss

@MODELS.register_module()
class RetinaYoloTalHead(AnchorHead):
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
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 share_head=False,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                init_cfg=dict(type='Kaiming', layer='Conv2d'), topk=13,**kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.share_head = share_head
        self.topk = topk
        super(RetinaYoloTalHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)
        print('TaskAlignedAssigner topk: {}'.format(self.topk))
        self.tal_assigner = TaskAlignedIgnoreAssigner(box_coder=self.bbox_coder, num_classes=self.num_classes,topk=self.topk)

    def _init_layers(self):
        """Initialize layers of the head."""
        # self.relu = nn.ReLU(inplace=True)
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
                        norm_cfg=self.norm_cfg))
                self.reg_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
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
                        norm_cfg=self.norm_cfg))
        
        self.retina_cls = nn.Conv2d(self.feat_channels,self.num_base_priors * self.cls_out_channels,3,padding=1)
        self.retina_reg = nn.Conv2d(self.feat_channels, self.num_base_priors * 4, 3, padding=1)
    
    def init_weights(self):
        """Initialize weights of the head."""

        if not self.share_head:
            for m in self.cls_convs:
                if isinstance(m, nn.Conv2d):
                    normal_init(m.conv, std=0.01, bias=bias_init_with_prob(0))
            for m in self.reg_convs:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01, bias=bias_init_with_prob(0))
        else:
            for m in self.shared_head_convs:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01, bias=bias_init_with_prob(0))

        normal_init(self.retina_cls, std=0.01, bias=bias_init_with_prob(0.01))
        normal_init(self.retina_reg, std=0.01, bias=bias_init_with_prob(0.01))

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

    # @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels, f'{len(featmap_sizes)}, {self.prior_generator.num_levels}'
        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        
        # try:
        labels, regression_targets, scores_targets, out_ignore_mask = self.tal_assigner(anchor_list, gt_bboxes, gt_labels, cls_scores, bbox_preds)
        # except Exception as e:
        #     return super().loss(cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore)

        N = len(labels)

        cls_scores, bbox_preds = concat_box_prediction_layers(cls_scores, bbox_preds)

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        scores_targets = torch.cat(scores_targets, dim=0)
        out_ignore_mask = torch.cat(out_ignore_mask, dim=0)

        pos_inds = (labels >= 0) & (labels != self.num_classes)
        num_pos_anchors = pos_inds.sum().item()

        # reg loss
        # self.bbox_coder.decode()
        if self.reg_decoded_bbox:
            anchors_over_all_imgs = torch.cat([torch.cat(anchors, dim=0) for anchors in anchor_list], dim=0)
            bbox_preds = self.bbox_coder.decode(anchors_over_all_imgs, bbox_preds)

        bbox_weights = torch.zeros_like(regression_targets)
        bbox_weights[pos_inds, :] = 1.
        losses_bbox = self.loss_bbox(
            bbox_preds,
            regression_targets,
            bbox_weights,
            avg_factor=(num_pos_anchors))

        if isinstance(self.loss_cls, YoloVariFocalLoss):
            losses_cls = self.loss_cls(cls_scores, labels, scores_targets, out_ignore_mask, avg_factor=(num_pos_anchors + N))
        else:
            losses_cls = self.loss_cls(cls_scores, labels, avg_factor=(num_pos_anchors + N))
       
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
    



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
    gt_bboxes_lt = _gt_bboxes[:, 0:2].unsqueeze(1).repeat(1, n_anchors, 1)
    gt_bboxes_rb = _gt_bboxes[:, 2:4].unsqueeze(1).repeat(1, n_anchors, 1)
    b_lt = xy_centers - gt_bboxes_lt
    b_rb = gt_bboxes_rb - xy_centers
    bbox_deltas = torch.cat([b_lt, b_rb], dim=-1)
    bbox_deltas = bbox_deltas.reshape([bs, n_max_boxes, n_anchors, -1])
    return (bbox_deltas.min(axis=-1)[0] > eps).to(gt_bboxes.dtype)

# def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
#     """if an anchor box is assigned to multiple gts,
#         the one with the highest iou will be selected.

#     Args:
#         mask_pos (Tensor): shape(bs, n_max_boxes, num_total_anchors)
#         overlaps (Tensor): shape(bs, n_max_boxes, num_total_anchors)
#     Return:
#         target_gt_idx (Tensor): shape(bs, num_total_anchors)
#         fg_mask (Tensor): shape(bs, num_total_anchors)
#         mask_pos (Tensor): shape(bs, n_max_boxes, num_total_anchors)
#     """
#     fg_mask = mask_pos.sum(axis=-2)
#     if fg_mask.max() > 1:
#         mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])
#         max_overlaps_idx = overlaps.argmax(axis=1)
#         is_max_overlaps = F.one_hot(max_overlaps_idx, n_max_boxes)
#         is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)
#         mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
#         fg_mask = mask_pos.sum(axis=-2)
#     target_gt_idx = mask_pos.argmax(axis=-2)
#     return target_gt_idx, fg_mask , mask_pos

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
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(
            box_cls_per_level, N, A, C, H, W
        )
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(
            box_regression_per_level, N, A, 4, H, W
        )
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = torch.cat(box_cls_flattened, dim=1).reshape(-1, C)
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression

class TaskAlignedIgnoreAssigner(nn.Module):
    def __init__(self,
                 box_coder,
                 topk=13,
                 num_classes=80,
                 alpha=1.0,
                 beta=6.0,
                 eps=1e-9):
        super(TaskAlignedIgnoreAssigner, self).__init__()
        self.init_topk = topk
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.box_coder = box_coder

    @torch.no_grad()
    # def forward(self,
    #             pd_scores,
    #             pd_bboxes,
    #             anc_points,
    #             gt_labels,
    #             gt_bboxes,
    #             mask_gt):
    def forward(self,
                anchors,
                targets_boxes_list,
                target_labels_list,
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
        box_cls_flattened = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_regression_flattened = [permute_to_N_HWA_K(x, 4) for x in box_regression]
        box_cls = torch.cat(box_cls_flattened, dim=1)
        box_regression = torch.cat(box_regression_flattened, dim=1)
        all_target_labels = []
        all_target_bboxes = []
        all_target_scores = []
        all_fg_mask = []
        all_ignore_mask = []
        # all_pred_score = []
        for anchors_per_image, targets_boxes, target_labels, box_cls_per_image, \
            box_regression_per_image in zip(anchors, targets_boxes_list, target_labels_list, box_cls, box_regression):
        
            anchors_over_all = torch.cat([anchor_per_image for anchor_per_image in anchors_per_image], dim=0)
            anchors_over_all_center = (anchors_over_all[:, :2] + anchors_over_all[:, 2:]) / 2
            # targets_boxes = targets_per_image.bbox
            # target_labels = targets_per_image.get_field("labels")

            bbox_pred_image = self.box_coder.decode(anchors_over_all, box_regression_per_image.detach())
            pd_scores = F.sigmoid(box_cls_per_image.detach()).unsqueeze(0)
            # _pd_scores = F.sigmoid(box_cls_per_image).unsqueeze(0)
            pd_bboxes = bbox_pred_image.unsqueeze(0)
            gt_bboxes = targets_boxes.unsqueeze(0)
            gt_labels = target_labels.unsqueeze(0).unsqueeze(-1)
            anc_points = anchors_over_all_center
            mask_gt = torch.ones(gt_labels.size()).to(anc_points.device)
        # print(pd_scores.size(), pd_bboxes.size(), gt_bboxes.size(), gt_labels.size(), anc_points.size(), mask_gt.size())
        
            self.bs = pd_scores.size(0)
            self.n_max_boxes = gt_bboxes.size(1)

            if self.n_max_boxes == 0:
                device = gt_bboxes.device
                target_labels = torch.full_like(pd_scores[..., 0], self.bg_idx).to(device)
                target_bboxes = torch.zeros_like(pd_bboxes).to(device)#.squeeze(0)
                target_scores = torch.zeros_like(pd_scores).to(device)
                fg_mask =  torch.zeros_like(pd_scores[..., 0]).to(device)
                ignore_mask =  torch.zeros_like(pd_scores[..., 0]).to(device)
            else:
                # print(pd_scores.size(), pd_bboxes.size(), gt_labels.size(), gt_bboxes.size(), anc_points.size())
                mask_pos, ignore_mask, align_metric, overlaps = self.get_pos_mask(
                    pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt)
                fg_mask = mask_pos.sum(axis=-2)
                ignore_mask = ignore_mask.sum(axis=-2)
                ignore_mask = ignore_mask == 0
                target_gt_idx = mask_pos.argmax(axis=-2)
                # target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(
                #     mask_pos, overlaps, self.n_max_boxes)

                # assigned target

                target_labels, target_bboxes, target_scores, target_gt_idx2 = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)
                target_labels = torch.where(fg_mask > 0, target_labels, torch.full_like(target_labels, self.num_classes))
                # normalize
                align_metric *= mask_pos
                pos_align_metrics = align_metric.max(axis=-1, keepdim=True)[0]
                pos_overlaps = (overlaps * mask_pos).max(axis=-1, keepdim=True)[0]
                norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).max(-2)[0].unsqueeze(-1)
                target_scores = target_scores * norm_align_metric
                # print(align_metric.size(), pos_overlaps.size(), pos_align_metrics.size(), target_scores.size(), norm_align_metric.size())
                # for i in range(target_labels.size(1)):
                #     if target_labels[0,i] == 10 and target_category_mask[0,i,8] == True:
                #         print(target_scores[0,i], target_category_mask[0,i])

                # print(target_bboxes.size())
                # target_bboxes = self.box_coder.encode(anchors_over_all, target_bboxes.squeeze(0))
                # print(target_bboxes.size())
        

        # print(all_target_labels.size(), all_target_bboxes.size(), all_target_scores.size(), all_fg_mask.size())

            # all_pred_score.append(_pd_scores)
            all_target_labels.append(target_labels)
            all_target_bboxes.append(target_bboxes)
            all_target_scores.append(target_scores)
            all_fg_mask.append(fg_mask.bool())
            all_ignore_mask.append(ignore_mask.bool())
            # print(target_labels.size(), target_bboxes.size(), target_scores.size(), fg_mask.size(), ignore_mask.size())

        # all_pred_score = torch.cat(all_pred_score, dim=0)
        all_target_labels = torch.cat(all_target_labels, dim=0).long()
        all_target_bboxes = torch.cat(all_target_bboxes, dim=0)
        all_target_scores = torch.cat(all_target_scores, dim=0)
        # print(all_target_labels.size(), all_target_bboxes.size(), all_target_scores.size())
        all_fg_mask = torch.cat(all_fg_mask, dim=0)
        all_ignore_mask = torch.cat(all_ignore_mask, dim=0)

        out_target_labels = []
        out_target_bboxes = []
        out_ignore_mask = []
        for i in range(all_target_labels.size(0)):
            out_target_labels.append(all_target_labels[i, :])
            out_target_bboxes.append(all_target_bboxes[i, :])
            out_ignore_mask.append(all_ignore_mask[i, :])


        out_target_scores = []
        for i in range(all_target_scores.size(0)):
            out_target_scores.append(all_target_scores[i, :])

        return out_target_labels, out_target_bboxes, out_target_scores, out_ignore_mask


    def get_pos_mask(self,
                     pd_scores,
                     pd_bboxes,
                     gt_labels,
                     gt_bboxes,
                     anc_points,
                     mask_gt):

        # get anchor_align metric
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)
        # print(align_metric.size(), overlaps.size(), pd_scores.size(), pd_bboxes.size(), gt_labels.size(), gt_bboxes.size())
        # get in_gts mask
        # print(anc_points, gt_bboxes)
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)
        # get topk_metric mask
        mask_topk = self.select_topk_candidates(align_metric * mask_in_gts, topk_mask=mask_gt.repeat([1, 1, self.topk]).bool())
        mask_top_one = self.select_topk_candidates(align_metric * mask_in_gts, topk_mask=mask_gt.repeat([1, 1, 1]).bool())
        # merge all mask to a final mask
        # print(mask_topk.size(), mask_in_gts.size(), mask_gt.size())
        # print(torch.sum(mask_topk), torch.sum(mask_in_gts), torch.sum(mask_gt))
        mask_pos = mask_top_one * mask_in_gts * mask_gt
        mask_ignore = (mask_topk * mask_in_gts * mask_gt) * (1-mask_pos)
        # print(torch.sum(mask_pos))

        return mask_pos, mask_ignore, align_metric, overlaps

    def get_box_metrics(self,
                        pd_scores,
                        pd_bboxes,
                        gt_labels,
                        gt_bboxes):

        pd_scores = pd_scores.permute(0, 2, 1)
        gt_labels = gt_labels.to(torch.long)
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1)
        bbox_scores = pd_scores[ind[0], ind[1]]

        overlaps = iou_calculator(gt_bboxes, pd_bboxes)
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)

        return align_metric, overlaps

    def select_topk_candidates(self,
                               metrics,
                               largest=True,
                               topk_mask=None):

        num_anchors = metrics.shape[-1]
        # print(metrics.size(), self.topk)
        topk_metrics, topk_idxs = torch.topk(
            metrics, self.topk, axis=-1, largest=largest)
        # print(topk_metrics.size(), topk_idxs.size())
        if topk_mask is None:
            topk_mask = (topk_metrics.max(axis=-1, keepdim=True) > self.eps).tile(
                [1, 1, self.topk])
        # print(topk_mask.device, topk_idxs.device, topk_metrics.device)
        topk_mask = topk_mask.to(topk_idxs.device)
        topk_idxs = torch.where(topk_mask, topk_idxs, torch.zeros_like(topk_idxs))
        # print(topk_idxs.size(), num_anchors)
        # print(F.one_hot(topk_idxs, num_anchors).size())
        for i in range(0, topk_idxs.size(1)):
            if i==0:
                topk_idxs_tmp = topk_idxs[:,i,:].unsqueeze(1)
                is_in_topk = F.one_hot(topk_idxs_tmp, num_anchors).sum(axis=-2)
            else:
                topk_idxs_tmp = topk_idxs[:,i,:].unsqueeze(1)
                is_in_topk_tmp = F.one_hot(topk_idxs_tmp, num_anchors).sum(axis=-2)
                is_in_topk = torch.cat([is_in_topk, is_in_topk_tmp],1)
        # is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(axis=-2)sd
        is_in_topk = torch.where(is_in_topk > 1,
            torch.zeros_like(is_in_topk), is_in_topk)
        return is_in_topk.to(metrics.dtype)

    def get_targets(self,
                    gt_labels,
                    gt_bboxes,
                    target_gt_idx,
                    fg_mask):

        # assigned target labels
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[...,None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
        target_labels = gt_labels.long().flatten()[target_gt_idx]

        # assigned target boxes
        target_bboxes = gt_bboxes.reshape([-1, 4])[target_gt_idx]

        # assigned target scores
        target_labels[target_labels<0] = 0
        target_scores = F.one_hot(target_labels, self.num_classes)
        fg_scores_mask  = fg_mask[:, :, None].repeat(1, 1, self.num_classes)
        target_scores = torch.where(fg_scores_mask > 0, target_scores,
                                        torch.full_like(target_scores, 0))

        return target_labels, target_bboxes, target_scores, target_gt_idx
