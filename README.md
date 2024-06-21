# Efficient Class Incremental Learning for Object Detection
- [Efficient Class Incremental Learning for Object Detection](#efficient-class-incremental-learning-for-object-detection)
- [Abstract](#abstract)
- [Get Started](#get-started)
  - [Installation](#installation)
  - [Download COCO](#download-coco)
- [Experiments](#experiments)
  - [COCO](#coco)
    - [40+40](#4040)
    - [70+10](#7010)
- [Training](#training)
  - [Normal training](#normal-training)
  - [CL Training](#cl-training)
- [Contributors](#contributors)

# Abstract
Object Detection, a critical task in computer vision, involves identifying and localizing items within an image. Continual Object Detection (COD) extends this by incrementally introducing training samples for different object categories, posing challenges due to limited access to past data and Catastrophic Forgetting. Traditional techniques like Knowledge Distillation and Exemplar Replay often fall short, and models with large parameters prolong training times, creating computational constraints. To address these issues, this study proposes the Efficient Continual Detection Transformer (ECOD), leveraging an efficient pretrained detector for generalization, pseudo-labeling for new data, and knowledge distillation on attention layers. LoRA optimizes parameter efficiency, reducing the parameters needed for fine-tuning while maintaining high performance. Extensive experiments on the COCO dataset validate this approach, demonstrating its superiority over state-of-the-art methods with only 3% of the trainable parameters, thus advancing the field of COD.
| ![Architecture](https://github.com/tuanlda78202/cod/blob/main/configs/slide.png) | 
|:--:| 
| Schematic of ECOD Framework|

  
# Get Started

## Installation 
```bash
git clone https://github.com/tuanlda78202/cod.git && cd cod
pip install -q -r requirements.txt
```
<!-- pipreqs for get requirements.txt -->

## Download COCO
```bash
mkdir coco && cd coco 
wget http://images.cocodataset.org/zips/train2017.zip && unzip train2017 && rm train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip & unzip val2017 && rm val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip & unzip annotations_trainval2017 && rm annotations_trainval2017.zip
cd ..
# Note: Change your COCO path on `configs/dataset/coco_detection.yml`
```
# Experiments

## COCO

### 40+40
| Method         | Baseline        | AP  | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> |
|:---------------|:----------------|------:|------:|------:|------:|------:|------:|
| LWF            | GFLv1           |  17.2 |  45.0 |  18.6 |   7.9 |  18.4 |  24.3 |
| RILOD          | GFLv1           |  29.9 |  45.0 |  32.9 |  18.5 |  33.0 |  40.5 |
| SID            | GFLv1           |  34.0 |  51.4 |  36.3 |  18.4 |  38.4 |  44.9 |
| ERD            | GFLv1           |  36.9 |  54.5 |  39.6 |  21.3 |  40.3 |  47.3 |
| CL-DETR        | Deformable DETR |  42.0 |  60.1 |  51.2 |  24.0 |  48.4 |  55.6 |
| SDDGR          | Deformable DETR |  43.0 |  62.1 |  47.1 |  24.9 |  46.9 |  57.0 |
| ECOD (Ours) | RT-DETR  |  **47.1** | **63.6** | **51.2** | **30.0** | **50.8** | **61.7** |
| Improv. (%)| -             | **9.53** | **2.42** | **8.70** | **20.48** | **8.32** | **8.25** |

### 70+10
| Method         | Baseline        | AP  | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> |
|:---------------|:----------------|------:|------:|------:|------:|------:|------:|
| LWF            | GFLv1           |  7.1  | 12.4  |  7.0  |  4.8  |  9.5  | 10.0  |
| RILOD          | GFLv1           | 24.5  | 37.9  | 25.7  | 14.2  | 27.4  | 36.4  |
| MMA            | -               | 30.2  | 52.1  | -     | -     | -     | -     |
| ABR            | -               | 31.1  | 52.9  | 32.7  | -     | -     | -     |
| SID            | GFLv1           | 32.8  | 49.9  | 35.0  | 17.1  | 36.9  | 44.5  |
| ERD            | GFLv1           | 34.9  | 51.9  | 35.7  | 17.4  | 38.8  | 45.4  |
| CL-DETR        | Deformable DETR | 35.8  | 53.5  | 39.5  | 19.4  | 43.0  | 48.6  |
| SDDGR          | Deformable DETR | 38.6  | 56.2  | 42.1  | 22.3  | 43.5  | 51.4  |
| VLM-PL         | Deformable DETR | 39.8  | 58.2  | 43.2  | 22.4  | 43.5  | 51.6  |
| ECL-DETR (Ours) | RT-DETR  |  **43.6** | **58.8** | **47.7** | **27.7** | **47.8** | **58.1** |
| Improv. (%) | -             | **9.55** | **1.03** | **10.42** | **23.66** | **9.89** | **12.60** |



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
  * `data_ratio` (`4040`, `7010`, `402020`, `4010101010`)
  * If CL, choose `task_idx` $\ge$ 1
  * If using buffer in CL, set `buffer_mode` = True and `buffer_rate`
* `configs/rtdetr/include/rtdetr_r50vd.yml`
  * In CL mode, set `task_idx` $\ge$ 1
*  `configs/cl_pipeline.yml`
   *  If using LoRA, set `lora_train` and `lora_val` to True
   *  `lora_cl`, `pseudo_label` and `distill_attn` set to True if CL 
   *  `teacher_path`: model trained on previous task 

```bash
# Note: Clean cache WandB
ps aux|grep wandb|grep -v grep | awk '{print $2}'|xargs kill -9
```

# Contributors 
<a href="https://github.com/tuanlda78202/cod/graphs/contributors">
<img src="https://contrib.rocks/image?repo=tuanlda78202/cod" /></a>
</a>
