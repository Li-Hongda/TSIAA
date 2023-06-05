_base_ = [
    'models/faster-rcnn_r50_fpn.py',
    'datasets/dior_detection.py',
    'schedules/schedule_2x.py', 'default_runtime.py'
]

# model settings
model = dict(roi_head=dict(bbox_head=dict(num_classes=20)))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize', scale=[(512, 512), (1024, 1024)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(pipeline=train_pipeline))

test_dataloader = dict(
    dataset=dict(
        data_root='/disk1/peileipl/datasets/DIOR/DIOR_2/allselect/',
        ann_file='select.json',
        data_prefix=dict(img='images/')))
test_evaluator = dict(
    type='CocoMetric',
    ann_file='/disk1/peileipl/datasets/DIOR/DIOR_2/allselect/select.json',
    metric='bbox',
    format_only=False,
    backend_args=None)

# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001))
optim_wrapper = dict(
    optimizer=dict(_delete_=True, type='AdamW', lr=0.00005, weight_decay=0.1),
    paramwise_cfg=dict(norm_decay_mult=0., bypass_duplicate=True))
