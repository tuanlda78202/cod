from typing import Iterable

import torch
import torch.amp

from src.data import CocoEvaluator
from src.misc import reduce_dict
from src.data.cococl import data_setting

from termcolor import colored, cprint
from pyexpat import model

import copy
import wandb
from tqdm import tqdm


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
        colored(
            f"Teacher Model loading complete from [{ckpt_path}]", "blue", "on_yellow"
        )
    )

    for _, params in model.named_parameters():
        params.requires_grad = False

    return model


def compute_attn(model, samples, targets, device, ex_device=None, mode="teacher"):
    if mode == "teacher":
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

            if ex_device is not None:
                model.to(ex_device)

    elif mode == "student":
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

    return model_encoder_outputs[0][-1]


def fake_query(outputs, targets, class_ids, topk=30, threshold=0.3):
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

    min_current_classes = min(class_ids)

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
    task_idx: int = None,
    data_ratio: str = None,
    pseudo_label: bool = None,
    distill_attn: bool = None,
    teacher_path: str = None,
    text_feat: torch.Tensor = None,
    prompt_mode: bool = True,
    **kwargs,
):
    model.train()
    criterion.train()

    if prompt_mode:
        model.backbone.eval()
        model.encoder.eval()

        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.encoder.parameters():
            param.requires_grad = False

        for name_p, p in model.decoder.named_parameters():
            if "prompt" in name_p:
                p.requires_grad = True
                print(name_p)
            else:
                p.requires_grad = False

    ema = kwargs.get("ema", None)
    scaler = kwargs.get("scaler", None)
    divided_classes = data_setting(data_ratio)

    if task_idx == 0:
        pseudo_label, distill_attn = False, False
        cprint("Normal Training...", "black", "on_yellow")

    if pseudo_label or distill_attn:
        teacher_copy = copy.deepcopy(model)
        teacher_model = load_model_params(teacher_copy, teacher_path)
        teacher_model.eval()

    tqdm_batch = tqdm(
        iterable=data_loader,
        desc="🚀 Epoch {}".format(epoch),
        total=len(data_loader),
        unit="it",
    )

    for _, (samples, targets, img_feats) in enumerate(tqdm_batch):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if distill_attn:
            teacher_attn = compute_attn(teacher_model, samples, targets, device)
            student_attn = compute_attn(model, samples, targets, device, mode="student")

            location_loss = torch.nn.functional.mse_loss(student_attn, teacher_attn)

            del teacher_attn, student_attn

        if pseudo_label:
            teacher_outputs = teacher_model(samples, targets)
            targets = fake_query(teacher_outputs, targets, divided_classes[task_idx])

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
            outputs = model(samples, targets, img_feats, text_feat)
            loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())

            if distill_attn:
                loss = loss + location_loss * 2

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        if ema is not None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        tqdm_batch.set_postfix(
            rtdetr_loss=loss_value.item(),
            kd_loss=location_loss.item() if distill_attn else 0,
            total_loss=loss.item() if distill_attn else 0,
        )

        wandb.log(
            {
                "RT-DETR Loss": loss_value,
                "KD Loss": (location_loss.item() if distill_attn else 0),
                "Total Loss": (loss.item() if distill_attn else 0),
            }
        )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessors,
    data_loader,
    base_ds,
    device,
    text_feat: torch.Tensor = None,
):
    model.eval()
    criterion.eval()

    iou_types = postprocessors.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    valid_tqdm_batch = tqdm(
        iterable=data_loader,
        desc="🏆 Valid ",
        total=len(data_loader),
        unit="it",
    )

    for batch_idx, (samples, targets, img_feats) in enumerate(valid_tqdm_batch):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples, image_query=img_feats, text_key=text_feat)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors(outputs, orig_target_sizes)

        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, results)
        }
        coco_evaluator.update(res)

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    stats = {}

    if "bbox" in iou_types:
        stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()

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

    return stats["coco_eval_bbox"][0] * 100
