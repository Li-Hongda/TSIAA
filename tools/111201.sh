CUDA_VISIBLE_DEVICES=2 python tools/attack.py \
configs/attack/faster-rcnn_r50_fpn_1x_dior.py \
work_dirs/faster-rcnn_r50_fpn_1x_dior/epoch_12.pth \
--attack tabim --name wce

CUDA_VISIBLE_DEVICES=2 python tools/attack.py \
configs/attack/atss_r50_fpn_1x_dior.py \
work_dirs/atss_r50_fpn_1x_dior/epoch_12.pth \
--attack tabim --name wce

CUDA_VISIBLE_DEVICES=2 python tools/attack.py \
configs/attack/retinanet_r50_fpn_1x_dior.py \
work_dirs/retinanet_r50_fpn_1x_dior/epoch_12.pth \
--attack tabim --name wce

CUDA_VISIBLE_DEVICES=2 python tools/attack.py \
configs/attack/fcos_r50-caffe_fpn_gn-head_1x_dior.py \
work_dirs/fcos_r50-caffe_fpn_gn-head_1x_dior/epoch_12.pth \
--attack tabim --name wce
