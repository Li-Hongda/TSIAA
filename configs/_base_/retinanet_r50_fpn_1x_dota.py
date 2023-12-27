_base_ = [
    'models/retinanet_r50_fpn.py',
    'datasets/dota_detection.py',
    'schedules/schedule_1x.py', 'default_runtime.py',
]

# model settings
model = dict(bbox_head=dict(num_classes=15))

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001))

test_dataloader = dict(
    dataset=dict(
        data_root='work_dirs/examples/dota/tsiaa_retinanet/',
        ann_file='select.json',
        data_prefix=dict(img='images/')))

test_evaluator = dict(type='ASRMetric', metric=['asr'])