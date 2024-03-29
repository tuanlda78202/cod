import math
import sys
from typing import Iterable

import torch
import torch.amp

from src.data import CocoEvaluator
from src.misc import MetricLogger, SmoothedValue, reduce_dict

from termcolor import colored
from pyexpat import model

import copy
import wandb


def load_model_params(model: model, ckpt_path: str = None):
    new_model_dict = model.state_dict()

    checkpoint = torch.load(ckpt_path)
    pretrained_model = checkpoint["model"]
    name_list = [
        name for name in new_model_dict.keys() if name in pretrained_model.keys()
    ]

    pretrained_model_dict = {
        k: v for k, v in pretrained_model.items() if k in name_list
    }

    new_model_dict.update(pretrained_model_dict)
    model.load_state_dict(new_model_dict)

    print(
        colored(f"Teacher Model loading complete from {ckpt_path}", "blue", "on_yellow")
    )

    for _, params in model.named_parameters():
        params.requires_grad = False

    return model


def compute_attn(model, samples, targets, device, ex_device):
    with torch.no_grad():
        model.to(device)

        model_encoder_outputs = []
        hook = (
            model.encoder.encoder[-1]
            .layers[-1]
            .self_attn.register_forward_hook(
                lambda module, input, output: model_encoder_outputs.append(output)
            )
        )

        _ = model(samples, targets)
        hook.remove()

        model.to(ex_device)

    return model_encoder_outputs[0][-1]


def fake_query(outputs, targets, current_classes, topk=30, threshold=0.3):
    out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

    prob = out_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(
        prob.view(out_logits.shape[0], -1), k=topk, dim=1
    )

    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    boxes = torch.gather(out_bbox, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
    results = [
        {"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)
    ]

    min_current_classes = min(current_classes)

    for idx, (target, result) in enumerate(zip(targets, results)):
        if target["labels"][target["labels"] < min_current_classes].shape[0] > 0:
            continue

        scores = result["scores"][result["scores"] > threshold].detach()
        labels = result["labels"][result["scores"] > threshold].detach()
        boxes = result["boxes"][result["scores"] > threshold].detach()

        if labels[labels < min_current_classes].size(0) > 0:
            addlabels = labels[labels < min_current_classes]
            addboxes = boxes[labels < min_current_classes]
            area = addboxes[:, 2] * addboxes[:, 3]

            targets[idx]["boxes"] = torch.cat((target["boxes"], addboxes))
            targets[idx]["labels"] = torch.cat((target["labels"], addlabels))
            targets[idx]["area"] = torch.cat((target["area"], area))
            targets[idx]["iscrowd"] = torch.cat(
                (target["iscrowd"], torch.tensor([0], device=torch.device("cuda")))
            )
    return targets


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    pseudo_label: bool = True,
    distill_attn: bool = True,
    teacher_path: str = "../detrw/4040_f40_10e_ap585.pth",
    current_classes=list(range(46, 91)),
    **kwargs,
):
    model.train()
    criterion.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = kwargs.get("print_freq", 10)

    ema = kwargs.get("ema", None)
    scaler = kwargs.get("scaler", None)

    if pseudo_label or distill_attn:
        device, ex_device = torch.device("cuda"), torch.device("cpu")
        teacher_copy = copy.deepcopy(model)
        student_copy = copy.deepcopy(model)

        teacher_model = load_model_params(teacher_copy, teacher_path)
        teacher_model.eval()

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if distill_attn:
            teacher_attn = compute_attn(
                teacher_model, samples, targets, device, ex_device
            )

            student_attn = compute_attn(
                student_copy, samples, targets, device, ex_device
            )

            location_loss = torch.nn.functional.mse_loss(student_attn, teacher_attn)

            del teacher_attn, student_attn

        if pseudo_label:
            teacher_model.to(device)
            teacher_outputs = teacher_model(samples, targets)
            targets = fake_query(teacher_outputs, targets, current_classes)

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets)

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())

            if distill_attn:
                loss = loss + location_loss * 0.5

            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        if ema is not None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        wandb.log(
            {
                "Total Loss": loss_value,
                "Loss VFL": loss_dict_reduced["loss_vfl"],
                "Loss GIoU": loss_dict_reduced["loss_giou"],
                "Loss BBox": loss_dict_reduced["loss_bbox"],
            }
        )

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessors,
    data_loader,
    base_ds,
    device,
    output_dir,
):
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    iou_types = postprocessors.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None

    for samples, targets in metric_logger.log_every(data_loader, 100, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors(outputs, orig_target_sizes)

        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, results)
        }
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}

    if coco_evaluator is not None:
        if "bbox" in iou_types:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in iou_types:
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    wandb.log(
        {
            "AP@0.5:0.95": stats["coco_eval_bbox"][0] * 100,
            "AP@0.5": stats["coco_eval_bbox"][1] * 100,
            "AP@0.75": stats["coco_eval_bbox"][2] * 100,
            "AP@0.5:0.95 Small": stats["coco_eval_bbox"][3] * 100,
            "AP@0.5:0.95 Medium": stats["coco_eval_bbox"][4] * 100,
            "AP@0.5:0.95 Large": stats["coco_eval_bbox"][5] * 100,
        }
    )
    return stats, coco_evaluator
