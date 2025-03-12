# dataset settings
dataset_type = 'MASADataset'
data_root = 'data/sam/'
img_scale = (1024, 576)

def get_dataset_dict_list(ann_file_list,
                         img_prefix_list,
                         dataset_type,
                         ):
    assert len(ann_file_list) == len(img_prefix_list)
    res_list = []
    for ann_file, img_prefix in zip(ann_file_list, img_prefix_list):
        item = dict(
        ann_file=ann_file,
        data_prefix=dict(img=img_prefix),
        type=dataset_type,
        serialize_data=True,
        metainfo={"classes": "body"},
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadMatchAnnotations'), ])
        res_list.append(item)

    return res_list


ann_file_list = [
    "/ssd/jfdeng/data/mall_train/mall_hk_rainy/annotations/instances_train.json",
    # "/face/jfdeng/data/mall_train/mall_scpg_rainy/annotations/instances_train.json",
    # "/face/jfdeng/data/mall_train/mall_taikooli_qt/annotations/instances_train_bhrf.json",
    # "/face/jfdeng/data/hqyc_train/annotations/instances_train_bhrf.json",
    # "/face/jfdeng/data/mall_train/mall_general_bhrf/annotations/instances_train.json",
    # "/face/jfdeng/data/mall_train/airport_sdjc_master/annotations/instances_train.json",
    # "/face/jfdeng/data/mall_train/airport_kmjc_bhrf/annotations/instances_train.json",
    # "/face/jfdeng/data/mall_train/mall_cmgj/annotations/instances_train_bhrf.json",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/annotations/instances_train_20211201_bottom_view_bhrf.json",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/annotations/instances_train_abc_0707_1_bhrf.json",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/annotations/instances_train_beijing_attr_bhrf.json",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/annotations/instances_train_cmcb_beijing_bhrf.json",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/annotations/instances_train_icbc_bhrf.json",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/annotations/instances_train_office_20220216_2_bhrf.json",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/annotations/instances_train_office_20220310_2_bhrf.json",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/annotations/instances_train_20211123_bottom_view_bhrf.json",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/annotations/instances_train_20211201_fid_bhrf.json",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/annotations/instances_train_abc_0707_2_bhrf.json",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/annotations/instances_train_ccb_bhrf.json",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/annotations/instances_train_ctf_2020_bhrf.json",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/annotations/instances_train_office_20220216_bhrf.json",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/annotations/instances_train_office_20220310_bhrf.json",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/annotations/instances_train_quanguo_attr_bhrf.json",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/annotations/instances_train_20211123_fid_bhrf.json",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/annotations/instances_train_abc_0707_bhrf.json",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/annotations/instances_train_abc_0707_3_bhrf.json",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/annotations/instances_train_cmcb_1217_bhrf.json",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/annotations/instances_train_GAC_bhrf.json",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/annotations/instances_train_office_20220216_1_bhrf.json",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/annotations/instances_train_office_20220310_1_bhrf.json",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/annotations/instances_train_part1_bhrf.json",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/annotations/instances_train_part2_bhrf.json",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/annotations/instances_train_origin_attr_bhrf.json",
    # "/face/ylzhang/data/bhrf_data/cj_20230610/instances_bhrf.json",
    # "/face/ylzhang/data/bhrf_data/fall_fight/data/Detection_MALL_beijing_office_20230517_ch/annotations/instaces.json",
    # "/face/ylzhang/data/bhrf_data/fall_fight/data/Detection_MALL_beijing_office_20230517_train/annotations/instances.json",
    # "/face/ylzhang/data/bhrf_data/mall_badcase_occ_20221222/annotations/instance_val.json",
    # "/face/ylzhang/data/bhrf_data/store_badcase_occ_20230206/annotations/instances.json",
    # "/face/ylzhang/data/bhrf_data/mall_chongqingxizhan_20230905/1/annotations/instances.json",
    # "/face/ykhu/BHRFD/data/bfj_det_train_data/train/annotations/air_person_20240605_bhrf.json",
    # "/face/ykhu/BHRFD/data/acdm_det_train_data/annotations/airport_bhrf_3class_images_train.json",
    # "/face/ykhu/BHRFD/data/bfj_det_train_data/train/annotations/hkpoc_20240926_bhrf.json",
    # "/face/ykhu/BHRFD/data/bfj_det_train_data/train/annotations/air_bj_20240911_bhrf.json",
    # "/face/ykhu/BHRFD/data/acdm_det_train_data/annotations/acdmbhrf_xm_20241128.json",
    #### "/face/ykhu/BHRFD/data/bfj_det_train_data/train/annotations/CrowdHuman_full_cocostyle_bhrf.json"

]
img_prefix_list = [
    "/ssd/jfdeng/data/mall_train/mall_hk_rainy/train",
    # "/face/jfdeng/data/mall_train/mall_scpg_rainy/train",
    # "/face/jfdeng/data/mall_train/mall_taikooli_qt/train",
    # "/face/jfdeng/data/hqyc_train/train",
    # "/face/jfdeng/data/mall_train/mall_general_bhrf/train",
    # "/face/jfdeng/data/mall_train/airport_sdjc_master/train",
    # "/face/jfdeng/data/mall_train/airport_kmjc_bhrf/train",
    # "/face/jfdeng/data/mall_train/mall_cmgj/train",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/train_20211201_bottom_view/",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/train_abc_0707_1/",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/train_beijing_attr/",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/train_cmcb_beijing/",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/train_icbc/",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/train_office_20220216_2/",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/train_office_20220310_2/",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/train_20211123_bottom_view/",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/train_20211201_fid/",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/train_abc_0707_2/",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/train_ccb/",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/train_ctf_2020/",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/train_office_20220216/",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/train_office_20220310/",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/train_quanguo_attr/",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/train_20211123_fid/",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/train_abc_0707/",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/train_abc_0707_3/",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/train_cmcb_1217/",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/train_GAC/",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/train_office_20220216_1/",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/train_office_20220310_1/",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/train_part1/",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/train_part2/",
    # "/face/xsqiu/dataset/store/store_train_self_training/dataset_for_pipeline_store_only_bhrf/train_origin_attr/",
    # "/face/ylzhang/data/bhrf_data/cj_20230610/imgs",
    # "/face/ylzhang/data/bhrf_data/fall_fight/data/Detection_MALL_beijing_office_20230517_ch/train/",
    # "/face/ylzhang/data/bhrf_data/fall_fight/data/Detection_MALL_beijing_office_20230517_train/train/",
    # "/face/ylzhang/data/bhrf_data/mall_badcase_occ_20221222/val",
    # "/face/ylzhang/data/bhrf_data/store_badcase_occ_20230206/train",
    # "/face/ylzhang/data/bhrf_data/mall_chongqingxizhan_20230905/1/imgs/",
    # "/face/ykhu/BHRFD/data/bfj_det_train_data/train/air_person_20240605",
    # "/face/ykhu/BHRFD/data/acdm_det_train_data/airport_bhrf_3class_images",
    # "/face/ykhu/BHRFD/data/bfj_det_train_data/train/hkpoc_20240926",
    # "/face/ykhu/BHRFD/data/bfj_det_train_data/train/air_bj_20240911",
    # "/face/ykhu/BHRFD/data/acdm_det_train_data/airport_bhrf_3class_images/acdmbhrf_xm_20241128",
    #### "/face/jfdeng/data/public/CrowdHuman_full_cocostyle/Images"
]

# data pipeline
train_pipeline = [
    dict(
        type='MixUniformRefFrameSample',
        num_ref_imgs=1,
        frame_range=0,
        filter_key_img=False),
    dict(
        type='MasaTransformBroadcaster',
        share_random_params=False,
        transforms=[
            dict(
                type='SeqRandomAffine',
            ),
            dict(
                type='SeqMixUp',
                img_scale=img_scale,
                ratio_range=(0.8, 1.6),
                pad_val=114.0,
                bbox_clip_border=False),
            dict(type='YOLOXHSVRandomAug'),
            dict(
                type='RandomResize',
                scale=img_scale,
                ratio_range=(0.1, 1.2),
                keep_ratio=True,
                clip_object_border=False),
            dict(type='RandomCrop', crop_size=img_scale, bbox_clip_border=False),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='Pad',
                size=img_scale,
                pad_val=114,
                # If the image is three-channel, the pad value needs
                # to be set separately for each channel
            ),
            # dict(type='SeqCopyPaste'),
            dict(type='FilterMatchAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
        ]),
    dict(type='PackMatchInputs')
]

test_pipeline = [
    dict(
        type='TransformBroadcaster',
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='LoadTrackAnnotations')
        ]),
    dict(type='PackTrackInputs')
]
train_dataset_dict_list = get_dataset_dict_list(ann_file_list,
                                            img_prefix_list,
                                            dataset_type,
                                            )

# dataloader
train_dataloader = dict(
    # batch_size=1,
    # num_workers=0,
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler'),  # image-based sampling
    dataset=dict(
        type='SeqMultiImageMixDataset',
        dataset=dict(
            type='RandomSampleConcatDataset',
            sampling_probs=[1],
            fixed_length=200,
            datasets=train_dataset_dict_list
    ),
        pipeline=train_pipeline

)
)

test_dataset_tpye = 'Taov1Dataset'

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    # Now we support two ways to test, image_based and video_based
    # if you want to use video_based sampling, you can use as follows
    sampler=dict(type='TrackImgSampler'),  # image-based sampling
    dataset=dict(
        type=test_dataset_tpye,
        ann_file='/ssd/yqshe/code/masa/data/tao/annotations/tao_val_lvis_v1_classes.json',
        data_prefix=dict(img_path='data/tao/frames/'),
        test_mode=True,
        pipeline=test_pipeline
    ))

test_dataloader = val_dataloader

# evaluator
val_evaluator = dict(
    type='TaoTETAMetric',
    dataset_type=test_dataset_tpye,
    format_only=False,
    ann_file='/ssd/yqshe/code/masa/data/tao/annotations/tao_val_lvis_v1_classes.json',
    metric=['TETA'])
test_evaluator = val_evaluator


