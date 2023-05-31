_base_ = [
    'models/faster-rcnn_r50_fpn.py',
    'datasets/dior_detection.py',
    'schedules/schedule_1x.py', 'default_runtime.py'
]

# model settings
model = dict(roi_head=dict(bbox_head=dict(num_classes=20)))

# test_dataloader = dict(
#     dataset=dict(
#         data_prefix=dict(
#             img='/disk2/lhd/codes/attack/work_dirs/examples/dior_tbim_fasterrcnn/images/')))

test_dataloader = dict(
    dataset=dict(
        data_root='/disk2/lhd/codes/attack/work_dirs/examples/dior_tmim_fasterrcnn/',
        ann_file='select.json',
        data_prefix=dict(img='images/')))
test_evaluator = dict(type='ASRMetric',
                      metric=['asr', 'dr'])
