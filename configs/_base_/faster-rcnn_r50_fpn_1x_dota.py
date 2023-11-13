_base_ = [
    'models/faster-rcnn_r50_fpn.py',
    'datasets/dota_detection.py',
    'schedules/schedule_1x.py', 'default_runtime.py'
]

model = dict(roi_head=dict(bbox_head=dict(num_classes=15)))

# test_dataloader = dict(
#     dataset=dict(
#         data_prefix=dict(
#             img='/disk2/lhd/codes/attack/work_dirs/examples/dota_bim_fasterrcnn_step10/')))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001))