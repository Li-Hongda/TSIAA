_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/dior_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(roi_head=dict(bbox_head=dict(num_classes=20)))

# model settings
runner_type = "AttackRunner"