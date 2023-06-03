_base_ = [
    'models/faster-rcnn_r50_fpn.py',
    'datasets/dior_detection.py',
    'schedules/schedule_2x.py', 'default_runtime.py'
]

# model settings
model = dict(roi_head=dict(bbox_head=dict(num_classes=20)))

# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001))
optim_wrapper = dict(
    optimizer=dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.1),
    paramwise_cfg=dict(norm_decay_mult=0., bypass_duplicate=True))
