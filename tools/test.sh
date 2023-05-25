#!/usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh
env=attack
conda activate ${env}

model=$1
attacker=$2
device=$3
show={4:-f}
data=${5:-dior}

if [ ${model} == fasterrcnn ]; then
    cfg=configs/_base_/faster-rcnn_r50_fpn_1x_${data}.py
    ckpt=work_dirs/faster-rcnn_r50_fpn_1x_${data}/epoch_12.pth
elif [ ${model} == fcos ]; then
    cfg=configs/_base_/fcos_r50-caffe_fpn_gn-head_1x_${data}.py
    ckpt=work_dirs/fcos_r50-caffe_fpn_gn-head_1x_${data}/epoch_12.pth
elif [ ${model} == retinanet ]; then
    cfg=configs/_base_/retinanet_r50_fpn_1x_${data}.py    
    ckpt=work_dirs/retinanet_r50_fpn_1x_${data}/epoch_12.pth
fi



opt=/disk2/lhd/codes/attack/work_dirs/examples/${data}_${attacker}_${model}_step10/

if [ ${show} == t ]; then
    CUDA_VISIBLE_DEVICES=${device} python tools/test.py ${cfg} ${ckpt} --cfg-options test_dataloader.dataset.data_prefix.img=${opt} --show-dir show
else 
    CUDA_VISIBLE_DEVICES=${device} python tools/test.py ${cfg} ${ckpt} --cfg-options test_dataloader.dataset.data_prefix.img=${opt}
fi


