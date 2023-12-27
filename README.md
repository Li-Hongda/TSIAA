# Task-Specific Importance-Awareness Matters: On Targeted Attacks against Object Detection




## Introduction

This is official implementation of paper "Task-Specific Importance-Awareness Matters: On Targeted Attacks against Object Detection".

Targeted Attacks on Object Detection aim to deceive the victim detector into recognizing a specific instance as the pre-defined target category while minimizing the changes to the predicted bounding box of that instance. Yet, this kind of flexible attack paradigm, which is capable of manipulating the decision outcome of the victim detector, received limited attention, especially in the context of OD in Optical Remote Sensing Images, where relevant research remains a blank. To fill this gap, this paper concentrates on targeted attacks on object detection in remote sensing images, and pays attention to a fundamental question, i.e., how to deploy targeted attacks on object detection via the raw predictions (that is, the predictions before non-maximum suppression) of a victim detector. In this regard, we depart from widely adopted task-independent importance measurements and hard-weighted ensemble optimization schemes present in existing methods. Instead, we first define the task-specific importance score, which considers both the quality and the attack costs of predictions. Next, we further propose a Task-Specific Importance- Aware Candidate Predictions Selection Scheme (TSIA-CPSS) alongside a Soft-Weighted Ensemble Optimization Scheme (SW-EOS).

## Getting Started
1. Clone the repo
2. Prepare environment with PyTorch
```
conda create -n tsiaa python=3.8
conda activate tsiaa
conda install pytorch torchvision -c pytorch
```
3. Install MMDetection
```
pip install -U openmim
mim install mmengine
mim install mmcv==2.0.0
cd /path/to/this/repo/
pip install -v -e .
```

## Datasets
The download link of these datasets can be accessed at [AAOD-ORSI](https://github.com/xuxiangsun/AAOD-ORSI).

## Citation
If you use our code/model/data, please cite our paper.

```
@article{
  title   = {Task-Specific Importance-Awareness Matters: On Targeted Attacks against Object Detection},
  author  = {Sun, Xuxiang, Cheng Gong, Li Hongda, Peng Hongyu and Han Junwei},
  journal = {arXiv preprint arXiv:},
   year   =  {2023}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

