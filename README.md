- [Get Started](#get-started)
  - [Installation](#installation)
  - [Download COCO](#download-coco)
- [RT-DETR Zoo](#rt-detr-zoo)
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

## Download COCO
```bash
mkdir coco && cd coco

wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip train2017 && rm train2017.zip
unzip val2017 && rm val2017.zip
unzip annotations_trainval2017 && rm annotations_trainval2017.zip
cd ..
```
Note: Change COCO path on `configs/dataset/coco_detection.yml`

# RT-DETR Zoo
| Model | Dataset | Input Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | #Params(M) | FPS |  checkpoint | O365 raw checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
rtdetr_18vd | COCO+Objects365 | 640 | 49.0 | 66.5 | 20 | 217 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth) | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_1x_objects365_from_paddle.pth)
rtdetr_r50vd | COCO+Objects365 | 640 | 55.2 | 73.4 | 42 | 108 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth) | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_1x_objects365_from_paddle.pth)
rtdetr_r101vd | COCO+Objects365 | 640 | 56.2 | 74.5 | 76 | 74 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_2x_coco_objects365_from_paddle.pth) | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_1x_objects365_from_paddle.pth)

Note: `COCO + Objects365` in the table means finetuned model on `COCO` using pretrained weights trained on `Objects365`.

# Training 
## Normal training
1. Training 
```bash
python scripts/train.py -t /path/to/ckpt/objects365
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
   2. Config `class_ids`, `buffer_ids` based on class order
   3. Set `True` for `fake_query` and `distill_attn`
   4. Training with same normal training script
   5. If you want evaluate, change `class_ids` to full classes

# Contributors 
<a href="https://github.com/tuanlda78202/MLR/graphs/contributors">
<img src="https://contrib.rocks/image?repo=tuanlda78202/MLR" /></a>
</a>