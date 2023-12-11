CUDA_VISIBLE_DEVICES=6 python tools/attack.py \
configs/attack/faster-rcnn_r50_fpn_1x_dota.py \
work_dirs/faster-rcnn_r50_fpn_1x_dota/epoch_12.pth \
--attack tbim --name tbim_fasterrcnn_random

CUDA_VISIBLE_DEVICES=6 python tools/attack.py \
configs/attack/atss_r50_fpn_1x_dota.py \
work_dirs/atss_r50_fpn_1x_dota/epoch_12.pth \
--attack tbim --name tbim_atss_random


CUDA_VISIBLE_DEVICES=6 python tools/attack.py \
configs/attack/fcos_r50-caffe_fpn_gn-head_1x_dota.py \
work_dirs/fcos_r50-caffe_fpn_gn-head_1x_dota/epoch_12.pth \
--attack tbim --name tbim_fcos_random


CUDA_VISIBLE_DEVICES=6 python tools/attack.py \
configs/attack/tood_r50_fpn_1x_dota.py \
work_dirs/tood_r50_fpn_1x_dota/epoch_12.pth \
--attack tbim --name tbim_tood_random

CUDA_VISIBLE_DEVICES=6 python tools/attack.py \
configs/attack/retinanet_r50_fpn_1x_dota.py \
work_dirs/retinanet_r50_fpn_1x_dota/epoch_12.pth \
--attack tbim --name tbim_retinanet_random

CUDA_VISIBLE_DEVICES=6 python tools/attack.py \
configs/attack/vfnet_r50_fpn_1x_dota.py \
work_dirs/vfnet_r50_fpn_1x_dota/epoch_12.pth \
--attack tbim --name tbim_vfnet_random

CUDA_VISIBLE_DEVICES=6 python tools/attack.py \
configs/attack/gfl_r50_fpn_1x_dota.py \
work_dirs/gfl_r50_fpn_1x_dota/epoch_12.pth \
--attack tbim --name tbim_gfl_random

