CUDA_VISIBLE_DEVICES=5 python tools/attack.py \
configs/attack/faster-rcnn_r50_fpn_1x_dior.py \
work_dirs/faster-rcnn_r50_fpn_1x_dior/epoch_12.pth \
--attack tcbim --name tcbim_fasterrcnn

CUDA_VISIBLE_DEVICES=5 python tools/attack.py \
configs/attack/atss_r50_fpn_1x_dior.py \
work_dirs/atss_r50_fpn_1x_dior/epoch_12.pth \
--attack tcbim --name tcbim_atss

CUDA_VISIBLE_DEVICES=5 python tools/attack.py \
configs/attack/retinanet_r50_fpn_1x_dior.py \
work_dirs/retinanet_r50_fpn_1x_dior/epoch_12.pth \
--attack tcbim --name tcbim_retinanet

CUDA_VISIBLE_DEVICES=5 python tools/attack.py \
configs/attack/fcos_r50-caffe_fpn_gn-head_1x_dior.py \
work_dirs/fcos_r50-caffe_fpn_gn-head_1x_dior/epoch_12.pth \
--attack tcbim --name tcbim_fcos

CUDA_VISIBLE_DEVICES=5 python tools/attack.py \
configs/attack/tood_r50_fpn_1x_dior.py \
work_dirs/tood_r50_fpn_1x_dior/epoch_12.pth \
--attack tcbim --name tcbim_tood

CUDA_VISIBLE_DEVICES=5 python tools/attack.py \
configs/attack/vfnet_r50_fpn_1x_dior.py \
work_dirs/vfnet_r50_fpn_1x_dior/epoch_12.pth \
--attack tcbim --name tcbim_vfnet

# CUDA_VISIBLE_DEVICES=5 python tools/attack.py \
# configs/attack/deformable-detr_r50_16xb2-50e_dior.py \
# work_dirs/deformable-detr_r50_16xb2-50e_dior/epoch.pth \
# --attack tcbim --name tcbim_