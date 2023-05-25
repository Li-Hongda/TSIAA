_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/debug_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(roi_head=dict(bbox_head=dict(num_classes=15)),
             test_cfg=dict(
                 rcnn=dict(
                    score_thr=0.05,
                    nms=dict(type='nms', iou_threshold=0.5),
                    max_per_img=100)))

# model settings
runner_type = "AttackRunner"

custom_hooks = [
    dict(
        type='RecordHook',
        output_dir = 'work_dirs/examples',
        priority=49)
]