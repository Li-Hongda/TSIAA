_base_ = [
    'models/faster-rcnn_r50_fpn.py',
    'datasets/dior_detection.py',
    'schedules/schedule_1x.py', 'default_runtime.py'
]

# model settings
model = dict(roi_head=dict(bbox_head=dict(num_classes=20)))


test_dataloader = dict(
    dataset=dict(
        data_root='work_dirs/examples/dior/tsiaa_fasterrcnn/',
        ann_file='select.json',
        data_prefix=dict(img='images/')))

test_evaluator = dict(type='ASRMetric', metric=['asr'])
