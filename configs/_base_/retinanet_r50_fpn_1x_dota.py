_base_ = [
    'models/retinanet_r50_fpn.py',
    'datasets/dior_detection.py',
    'schedules/schedule_1x.py', 'default_runtime.py',
    './retinanet_tta.py'
]

# model settings
model = dict(bbox_head=dict(num_classes=15))

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
