_base_ = '/cm/shared/user/Project/CPA-Enhancer/configs/_base_/default_runtime.py'
pretrained = '/cm/shared/user/Project/CPA-Enhancer/pretrained_weight/resnet18vd_pretrained_55f5a0d6.pth'  # change this to pretrained_weight path or download 
# https://github.com/flytocc/mmdetection/releases/download/model_zoo/resnet18vd_pretrained_265f1124.pth

backend_args = None
resume = False

data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[0, 0, 0],
    std=[255., 255., 255.],
    bgr_to_rgb=True,
    pad_size_divisor=32)

model = dict(
    type='FRDETR',
    num_queries=300,  # num_matching_queries, 900 for DINO
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='ResNetV1d',  # ResNet for DINO
        depth=18,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,  # -1 for DINO
        norm_cfg=dict(type='SyncBN', requires_grad=True),  # BN for DINO
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='ChannelMapper',
        in_channels=[128, 256, 512],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='SyncBN', requires_grad=True),  # GN for DINO
        num_outs=3),  # 4 for DINO
    encoder=dict(
        use_encoder_idx=[2],
        num_encoder_layers=1,
        in_channels=[256, 256, 256],
        fpn_cfg=dict(
            type='RTDETRFPN',
            in_channels=[256, 256, 256],
            out_channels=256,
            expansion=0.5,
            norm_cfg=dict(type='SyncBN', requires_grad=True)),
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=1024,  # 2048 for DINO
                ffn_drop=0.0,
                act_cfg=dict(type='GELU')))),  # ReLU for DINO
        decoder=dict(
        num_layers=3,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(
                embed_dims=256,
                num_levels=3,  # 4 for DINO
                dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=1024,  # 2048 for DINO
                ffn_drop=0.0)),
        post_norm_cfg=None),
    bbox_head=dict(
        type='RTDETRHead',
        num_classes=5,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='RTDETRVarifocalLoss',  # FocalLoss in DINO
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries for sparse VOC dataset
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=100)) # TODO: might have to change to much lower like 100 or 50

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=[0, 0, 0],
        to_rgb=True,
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', scale=(448, 448), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(448, 448), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# optimizer
# optimizer
num_blocks_list = (2, 2, 2, 2)  # r18
downsample_norm_idx_list = (2, 3, 3, 3)  # r18
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
paramwise_cfg = dict(
    custom_keys={'backbone': dict(lr_mult=0.1)},
    norm_decay_mult=0,
    bypass_duplicate=True
)
# Update custom_keys with backbone normalization keys
paramwise_cfg['custom_keys'].update({
    f'backbone.layer{stage_id + 1}.{block_id}.bn': backbone_norm_multi
    for stage_id, num_blocks in enumerate(num_blocks_list)
    for block_id in range(num_blocks)
})
paramwise_cfg['custom_keys'].update({
    f'backbone.layer{stage_id + 1}.{block_id}.downsample.{downsample_norm_idx - 1}':  # noqa
    backbone_norm_multi
    for stage_id, (num_blocks, downsample_norm_idx) in enumerate(
        zip(num_blocks_list, downsample_norm_idx_list))
    for block_id in range(num_blocks)
})
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=paramwise_cfg)

# learning policy
max_epochs = 48
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=2000)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]


dataset_type = 'VOCDataset'
data_root = '/cm/shared/user/data/cpa_data'

train_dataloader = dict(
    batch_size=6,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="/cm/shared/user/data/cpa_data/voc_hybrid_fog/train/ImageSets/Main/train_voc.txt",
        data_prefix=dict(sub_data_root="/cm/shared/user/data/cpa_data/voc_hybrid_fog/train"),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="/cm/shared/user/data/cpa_data/voc_fog/test/ImageSets/Main/test_voc.txt",
        data_prefix=dict(sub_data_root="/cm/shared/user/data/cpa_data/voc_fog/test"),
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="/cm/shared/user/data/cpa_data/voc_fog/test/ImageSets/Main/test_voc.txt",
        data_prefix=dict(sub_data_root="/cm/shared/user/data/cpa_data/voc_fog/test"),
        # ann_file="/cm/shared/user/data/cpa_data/voc_norm_5cls/test/ImageSets/Main/test_voc.txt",
        # data_prefix=dict(sub_data_root="/cm/shared/user/data/cpa_data/voc_norm_5cls/test"),
        # ann_file="/cm/archive/user/data/RTTS/ImageSets/Main/test.txt",
        # data_prefix=dict(sub_data_root="/cm/archive/user/data/RTTS"),
        test_mode=True,
        pipeline=test_pipeline))

val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points', iou_thrs=[0.25, 0.5, 0.75])
test_evaluator = val_evaluator

# Checkpoint saving
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=200),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

visualizer = dict(vis_backends = [
                                  dict(
                                    type='WandbVisBackend',
                                    init_kwargs={
                                        'settings': dict(
                                            _disable_stats=True
                                        )
                                    },),
                                  ])
