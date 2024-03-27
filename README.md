- [Get Started](#get-started)
  - [Installation](#installation)
  - [Download COCO Dataset](#download-coco-dataset)
- [Training](#training)
  - [Normal training](#normal-training)
  - [CL Training](#cl-training)
  - [Contributors](#contributors)
  
# Get Started

## Installation 
1. Clone the repository
```bash
git clone https://github.com/tuanlda78202/cod.git cod && cd cod
```
2. Install the required packages:
```
pip install -r requirements.txt
```
<!-- pipreqs for get requirements.txt -->

## Download COCO Dataset
```bash
mkdir coco
%cd coco

wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip train2017 && rm train2017.zip
unzip val2017 && rm val2017.zip
unzip annotations_trainval2017 && rm annotations_trainval2017.zip
cd ..
```
Note: Change COCO path on `configs/dataset/coco_detection.yml`

# Training 
## Normal training
1. Training 
```bash
python scripts/train.py -t /path/to/ckpt
```
2. Evaluate 
```bash
python scripts/train.py -r /path/to/ckpt/training --test-only
```

## CL Training 
1. Choose data ratio (`4040`, `7010`)
2. Normal training for first task `t` classes 
3. For next task
   1. Load checkpoint trained from previous task (manual)
   2. Config `class_ids`, `buffer_ids` on `coco_cache.py`
   3. Set `True` for `fake_query` and `distill_attn`
   4. Training with same scripts above

## Contributors 
<a href="https://github.com/tuanlda78202/MLR/graphs/contributors">
<img src="https://contrib.rocks/image?repo=tuanlda78202/MLR" /></a>
</a>