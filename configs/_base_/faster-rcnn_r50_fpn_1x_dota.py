_base_ = [
    'models/faster-rcnn_r50_fpn.py',
    'datasets/dota_detection.py',
    'schedules/schedule_1x.py', 'default_runtime.py'
]

model = dict(roi_head=dict(bbox_head=dict(num_classes=15)))


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001))

test_dataloader = dict(
    dataset=dict(
        data_root='/disk2/lhd/codes/attack/work_dirs/examples/dota/tbim_fasterrcnn_test/',
        ann_file='select.json',
        data_prefix=dict(img='images/')))

test_evaluator = dict(type='ASRMetric',
                      metric=['asr', 'dr'])