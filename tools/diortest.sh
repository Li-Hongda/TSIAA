CUDA_VISIBLE_DEVICES=4 python tools/test.py \
configs/_base_/faster-rcnn_r50_fpn_1x_dior.py \
work_dirs/faster-rcnn_r50_fpn_1x_dior/epoch_12.pth

CUDA_VISIBLE_DEVICES=4 python tools/test.py \
configs/_base_/atss_r50_fpn_1x_dior.py \
work_dirs/atss_r50_fpn_1x_dior/epoch_12.pth

CUDA_VISIBLE_DEVICES=4 python tools/test.py \
configs/_base_/fcos_r50-caffe_fpn_gn-head_1x_dior.py \
work_dirs/fcos_r50-caffe_fpn_gn-head_1x_dior/epoch_12.pth

# CUDA_VISIBLE_DEVICES=4 python tools/test.py \
# configs/_base_/tood_r50_fpn_1x_dior.py \
# work_dirs/tood_r50_fpn_1x_dior/epoch_12.pth

# CUDA_VISIBLE_DEVICES=4 python tools/test.py \
# configs/_base_/retinanet_r50_fpn_1x_dior.py \
# work_dirs/retinanet_r50_fpn_1x_dior/epoch_12.pth

CUDA_VISIBLE_DEVICES=4 python tools/test.py \
configs/_base_/vfnet_r50_fpn_1x_dior.py \
work_dirs/vfnet_r50_fpn_1x_dior/epoch_12.pth

# CUDA_VISIBLE_DEVICES=4 python tools/test.py \
# configs/_base_/gfl_r50_fpn_1x_dior.py \
# work_dirs/gfl_r50_fpn_1x_dior/epoch_12.pth