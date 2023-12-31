_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/dota_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(roi_head=dict(bbox_head=dict(num_classes=15)))

# model settings
runner_type = "AttackRunner"

custom_hooks = [
    dict(
        type='RecordHook',
        output_dir = 'work_dirs/examples/dota',
        priority=49)
]