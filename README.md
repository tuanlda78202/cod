# Efficient Detection Transformer for Incremental Object Detection
- [Efficient Detection Transformer for Incremental Object Detection](#efficient-detection-transformer-for-incremental-object-detection)
- [Get Started](#get-started)
  - [Installation](#installation)
  - [Download COCO](#download-coco)
- [Experiments](#experiments)
  - [CL COCO](#cl-coco)
    - [40-40](#40-40)
    - [70-10](#70-10)
  - [RT-DETR Zoo](#rt-detr-zoo)
- [Training](#training)
  - [Normal training](#normal-training)
  - [CL Training](#cl-training)
- [Contributors](#contributors)

<div style="border:1px solid black; padding:10px; margin-top:5px;">
  In the realm of object detection, both incremental and real-time challenges necessitate advanced strategies to improve performance without sacrificing speed or succumbing to catastrophic forgetting. In this paper, we introduce innovative methods for addressing challenges in object detection using transformer-based models. The Continual Detection Transformer (CL-DETR) enhances incremental object detection by incorporating Detector Knowledge Distillation (DKD) and an exemplar replay calibration strategy to mitigate catastrophic forgetting. Simultaneously, the Real-Time Detection Transformer (RT-DETR) optimizes for speed and accuracy in real-time scenarios through a hybrid encoder and IoU-aware query selection, eliminating the need for non-maximum suppression. Additionally, we explore the efficiency of Low-Rank Adaptation (LoRA), which significantly reduces trainable parameters in large-scale models by integrating trainable rank decomposition matrices, allowing for cost-effective adaptation without extensive retraining. These approaches collectively advance the performance and practicality of transformer-based object detection systems.
</div>
  
# Get Started

<details>
<summary>Folder Structure</summary>
├── CODEOWNERS
├── configs
│   ├── cl_pipeline.yml
│   ├── dataset
│   │   └── coco_detection.yml
│   ├── rtdetr
│   │   ├── include
│   │   │   ├── dataloader.yml
│   │   │   ├── optimizer.yml
│   │   │   └── rtdetr_r50vd.yml
│   │   └── rtdetr_r50vd_coco.yml
│   └── runtime.yml
├── LICENSE
├── README.md
├── requirements.txt
├── scripts
│   └── train.py
└── src
    ├── core
    │   ├── config.py
    │   ├── __init__.py
    │   ├── yaml_config.py
    │   └── yaml_utils.py
    ├── data
    │   ├── cococl
    │   │   ├── cl_utils.py
    │   │   ├── coco_cache.py
    │   │   ├── coco_cl.py
    │   │   ├── coco_eval.py
    │   │   ├── coco_utils.py
    │   │   ├── custom_coco_eval.py
    │   │   └── __init__.py
    │   ├── dataloader.py
    │   ├── functional.py
    │   ├── __init__.py
    │   └── transforms.py
    ├── __init__.py
    ├── misc
    │   ├── dist.py
    │   ├── __init__.py
    │   ├── logger.py
    │   └── visualizer.py
    ├── nn
    │   ├── arch
    │   │   ├── classification.py
    │   │   └── __init__.py
    │   ├── backbone
    │   │   ├── common.py
    │   │   ├── __init__.py
    │   │   ├── presnet.py
    │   │   ├── test_resnet.py
    │   │   └── utils.py
    │   ├── criterion
    │   │   ├── __init__.py
    │   │   └── utils.py
    │   └── __init__.py
    ├── optim
    │   ├── amp.py
    │   ├── ema.py
    │   ├── __init__.py
    │   └── optim.py
    ├── rtdetr
    │   ├── box_ops.py
    │   ├── denoising.py
    │   ├── hybrid_encoder.py
    │   ├── __init__.py
    │   ├── matcher.py
    │   ├── rtdetr_criterion.py
    │   ├── rtdetr_decoder.py
    │   ├── rtdetr_postprocessor.py
    │   ├── rtdetr.py
    │   └── utils.py
    └── solver
        ├── det_engine.py
        ├── det_solver.py
        ├── __init__.py
        └── solver.py
</details>



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
Note: Change COCO path on `configs/dataset/coco_detection.yml`

# Experiments
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

## RT-DETR Zoo
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