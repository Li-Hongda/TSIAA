_base_ = [
    'models/retinanet_r50_fpn.py',
    'datasets/dior_detection.py',
    'schedules/schedule_1x.py', 'default_runtime.py',
    # '../retinanet/retinanet_tta.py'
]

# model settings
model = dict(bbox_head=dict(num_classes=20))

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

test_dataloader = dict(
    dataset=dict(
        data_prefix=dict(
            img='/disk2/lhd/codes/attack/work_dirs/examples/dior_bim_fasterrcnn_step10/')))