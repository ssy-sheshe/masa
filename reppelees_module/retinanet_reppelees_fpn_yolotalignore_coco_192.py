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
        pretrained='/face/ykhu/BHRFD/git/mmdetection/resources/reppelees_cpu_70.01.pth'
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

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1024, 576), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape','scale_factor'))
]

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='Airport',
        data_root='/',
        ann_file='/face/ykhu/BHRFD/data/acdm_det_train_data/annotations/air_general_bmk_min.coco',
        data_prefix=dict(img='/face/ykhu/BHRFD/data/acdm_det_train_data/airport_object_8class_images/air_general_bmk_min/'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

evaluation = dict(interval=1, metric='bbox')
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/face/ylzhang/retinanet_reppelees_fpn_object365/epoch_37.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
optimizer = dict(
    type='AdamW', lr=0.0004, betas=(0.9, 0.999), weight_decay=0.05)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    by_epoch=False,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=1e-06,
    min_lr=0)
runner = dict(type='EpochBasedRunner', max_epochs=100)
work_dir = '/face/ykhu/BHRFD/git/log/mmdetection_work_dir/retinanet_reppelees_fpn_yolotalignore_coco_192_datasetv13_mean0_std255_bgr'
auto_resume = False
gpu_ids = range(0, 8)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')