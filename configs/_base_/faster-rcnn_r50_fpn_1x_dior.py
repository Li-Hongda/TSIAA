_base_ = [
    'models/faster-rcnn_r50_fpn.py',
    'datasets/dior_detection.py',
    'schedules/schedule_1x.py', 'default_runtime.py'
]

# model settings
model = dict(roi_head=dict(bbox_head=dict(num_classes=20)),
             test_cfg=dict(
                 rcnn=dict(
                    score_thr=0.3,
                    nms=dict(type='nms', iou_threshold=0.5),
                    max_per_img=100)))

test_dataloader = dict(
    dataset=dict(
        data_prefix=dict(
            img='/disk2/lhd/codes/attack/work_dirs/examples/dior_hbb_bimiou_fasterrcnn_step10/')))
