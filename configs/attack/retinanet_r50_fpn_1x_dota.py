_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/dota_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
]

# model settings
model = dict(bbox_head=dict(num_classes=15))

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001))

runner_type = "AttackRunner"
custom_hooks = [
    dict(
        type='RecordHook',
        output_dir = 'work_dirs/examples',
        priority=49)
]
