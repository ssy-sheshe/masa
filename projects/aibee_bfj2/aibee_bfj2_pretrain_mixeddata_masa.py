# _base_ = [
#     './grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py',
# ]
#
# keys_to_delete = [key for key in _base_.keys() if key != 'model']
# for key in keys_to_delete:
#     del _base_[key]


model = dict(
    backbone=dict(
        freeze_backbone=False,
        freeze_bn=False,
        norm_cfg=dict(type='SyncBN'),
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        pretrained=
        '/ssd/yqshe/code/masa/saved_models/pretrain_weights/reppelees_cpu_70.01.pth',
        type='RepPeleeNetS'),
    bbox_head=dict(
        anchor_generator=dict(
            anchor_sizes=[
                20,
                40,
                80,
                160,
            ],
            anchor_strides=[
                8,
                16,
                32,
                64,
            ],
            aspect_ratios=[
                1.0,
                2.5,
                3.5,
            ],
            octave_base_scale=2.0,
            scales_per_octave=3,
            type='CrystalAnchorGenerator'),
        bbox_coder=dict(
            type='CrystalBoxCoder', weights=[
                10.0,
                10.0,
                5.0,
                5.0,
            ]),
        feat_channels=192,
        in_channels=192,
        loss_bbox=dict(loss_weight=4.0, type='CIoULoss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        loss_hook_bbox=dict(loss_weight=2.0, type='CIoULoss'),
        mode='train',
        num_classes=2,
        reg_decoded_bbox=True,
        share_head=True,
        stacked_convs=4,
        topk=13,
        type='BFJYoloTalOTOHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        add_extra_convs='on_input',
        in_channels=[
            128,
            256,
            384,
            512,
        ],
        num_outs=4,
        out_channels=192,
        start_level=1,
        type='FPN'),
    test_cfg=dict(
        max_per_img=1000,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.8, type='nms'),
        nms_pre=1000,
        score_thr=0.01),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(
            ignore_iof_thr=-1,
            min_pos_iou=0,
            neg_iou_thr=0.4,
            pos_iou_thr=0.5,
            type='MaxIoUAssigner'),
        debug=False,
        pos_weight=-1,
        sampler=dict(type='PseudoSampler')),
    type='RetinaNet')