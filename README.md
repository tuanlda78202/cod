# Efficient Class Incremental Learning for Object Detection
- [Efficient Class Incremental Learning for Object Detection](#efficient-class-incremental-learning-for-object-detection)
- [Abstract](#abstract)
- [Get Started](#get-started)
  - [Installation](#installation)
  - [Download COCO](#download-coco)
- [Experiments](#experiments)
  - [Zoo](#zoo)
  - [CL COCO](#cl-coco)
    - [40-40](#40-40)
    - [70-10](#70-10)
- [Training](#training)
  - [Normal training](#normal-training)
  - [CL Training](#cl-training)
- [Contributors](#contributors)

# Abstract
In the realm of object detection, both incremental and real-time challenges necessitate advanced strategies to improve performance without sacrificing speed or succumbing to catastrophic forgetting. In this paper, we introduce innovative methods for addressing challenges in object detection using transformer-based models. The Continual Detection Transformer (CL-DETR) enhances incremental object detection by incorporating Detector Knowledge Distillation (DKD) and an exemplar replay calibration strategy to mitigate catastrophic forgetting. Simultaneously, the Real-Time Detection Transformer (RT-DETR) optimizes for speed and accuracy in real-time scenarios through a hybrid encoder and IoU-aware query selection, eliminating the need for non-maximum suppression. Additionally, we explore the efficiency of Low-Rank Adaptation (LoRA), which significantly reduces trainable parameters in large-scale models by integrating trainable rank decomposition matrices, allowing for cost-effective adaptation without extensive retraining. These approaches collectively advance the performance and practicality of transformer-based object detection systems.
| ![Architecture](https://github.com/tuanlda78202/cod/blob/main/configs/slide.png) | 
|:--:| 
| Schematic of ECOD Framework|

  
# Get Started

## Installation 
1. Clone the repository
```bash
git clone https://github.com/tuanlda78202/cod.git && cd cod
```
2. Install the required packages:
```
pip install -q -r requirements.txt
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
Note: Change your COCO path on `configs/dataset/coco_detection.yml`

# Experiments
## Zoo
| Model          | Dataset        | Input Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | #Params(M) | FPS | checkpoint | O365 raw checkpoint |
| :------------: | :------------: | :--------: | :--------------: | :--------------------------: | :--------: | :--: | :--------: | :-----------------: |
| rtdetr_r50vd   | COCO+Objects365| 640        | 55.2             | 73.4                         | 42         | 108  | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth) | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_1x_objects365_from_paddle.pth) |

## CL COCO

### 40-40
| Setting | Method   | Technique                  | Task         | AP  | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> |
|---------|----------|----------------------------|--------------|-----|------|-------|------|------|------|
| 40+40   | CL-DETR  | Incremental Learning       | Eval all data| 42  | 60.1 | 45.9  | 24   | 45.3 | 55.6 |
| 40+40   | SDDGR    | Incremental Learning       | Eval all data| 43  | 62.1 | 47.1  | 24.9 | 46.9 | 57   |
| 40+40   | CL-RT-DETR | Fake Query + Distill Attn. + Buffer 10% | Eval all data   | 48.7| 65.1 | 53.1  | 31.4 | 52.7 | 63.1 |
### 70-10
| Setting | Method   | Technique                  | Task         | AP  | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> |
|---------|------------|---------------------------------------------------|----------------------|------|------|-------|------|------|------|
| 70+10   | CL-DETR    | Incremental Learning                              | Eval all data        | 40.4 | 58   | 43.9  | 23.8 | 43.6 | 53.5 |
| 70+10   | SDDGR      | Incremental Learning                              | Eval all data        | 40.9 | 59.5 | 44.8  | 23.9 | 44.7 | 54   |
| 70+10   | CL-RT-DETR | Fake Query + Distill Attn. + Buffer 10%           | Eval all data (1e task 2) | 44.4 | 59.6 | 48.2  | 28.7 | 48.5 | 60   |


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
* `configs/rtdetr/include/dataloader.yml`
  * `data_ratio` (`4040`, `7010`)
  * If CL, choose `task_idx` = 1
  * If using buffer in CL, set `buffer_mode` = True and `buffer_rate`
* `configs/rtdetr/include/rtdetr_r50vd.yml`
  * In CL mode, set `task_idx` = 1
*  `configs/cl_pipeline.yml`
   *  If using LoRA, set `lora_train` and `lora_val` to True
   *  `lora_cl`, `pseudo_label` and `distill_attn` set to True if CL 
   *  `teacher_path`: model trained on previous task 

Note: Clean cache WandB
```bash
ps aux|grep wandb|grep -v grep | awk '{print $2}'|xargs kill -9
```

# Contributors 
<a href="https://github.com/tuanlda78202/MLR/graphs/contributors">
<img src="https://contrib.rocks/image?repo=tuanlda78202/MLR" /></a>
</a>