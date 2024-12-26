# _base_ = [
#     './grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py',
# ]
#
# keys_to_delete = [key for key in _base_.keys() if key != 'model']
# for key in keys_to_delete:
#     del _base_[key]


model = dict(
    type='RetinaNet',
        data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[0, 0, 0],
        std=[255, 255, 255],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='RepPeleeNetS',
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(
            type='SyncBN',
            requires_grad=True,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True),
        repvgg_converted = False,
        pretrained='/ssd/yqshe/code/masa/saved_models/pretrain_weights/reppelees_cpu_70.01.pth'
        ),
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 384, 512],
        out_channels=192,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=4),
    bbox_head=dict(
        type='RetinaYoloTalHead',
        num_classes=8,
        in_channels=192,
        stacked_convs=4,
        feat_channels=192,
        share_head=True,
        reg_decoded_bbox=True,
        anchor_generator=dict(
            type='CrystalAnchorGenerator',
            anchor_sizes=[16, 56, 120, 256],
            octave_base_scale=2.0,
            scales_per_octave=3,
            aspect_ratios=[0.5, 1.0, 2.0],
            anchor_strides=[8, 16, 32, 64]),
        bbox_coder=dict(
            type='CrystalBoxCoder', weights=[10.0, 10.0, 5.0, 5.0]),
        loss_cls=dict(
            type='YoloVariFocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.75,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.5)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.01,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=1000))