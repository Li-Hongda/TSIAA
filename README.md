# Task-Specific Importance-Awareness Matters: On Targeted Attacks against Object Detection


## Datasets
The download link of these datasets can be accessed at [AAOD-ORSI](https://github.com/xuxiangsun/AAOD-ORSI).

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
4. Run 
attack
```
python tools/attack.py /path/to/config /path/to/checkpoint --name /save/name/ \
--epsilon 16 --steps 20
```
test 
```
python tools/test.py /path/to/config /path/to/checkpoint 
```


## License

This project is released under the [Apache 2.0 license](LICENSE).

