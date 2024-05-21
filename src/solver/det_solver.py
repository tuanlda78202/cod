from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate

from termcolor import cprint
from peft import LoraConfig, get_peft_model, PeftModel
import torch


class DetSolver(BaseSolver):

    def fit(
        self,
    ):
        self.train()

        args = self.cfg
        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        task_idx = self.train_dataloader.dataset.task_idx
        data_ratio = self.train_dataloader.dataset.data_ratio

        if args.lora_train:
            if not args.lora_cl:
                lora_modules = [
                    name
                    for name, module in self.model.named_modules()
                    if any(
                        layer in str(type(module))
                        for layer in ["Linear", "linear", "Conv2d", "Embedding"]
                    )
                    and "Identity" not in str(type(module))
                ]
                config = LoraConfig(target_modules=lora_modules)
                self.lora_model = get_peft_model(self.model, config)
            else:
                self.lora_model = PeftModel.from_pretrained(
                    self.model, args.teacher_path, is_trainable=True
                )

            self.lora_model.print_trainable_parameters()

            lora_params = [
                {
                    "params": [
                        p
                        for n, p in getattr(self.lora_model, part).named_parameters()
                        if "lora" in n
                    ],
                    **params,
                }
                for part, params in zip(
                    ["backbone", "encoder", "decoder"],
                    [{"lr": 0.00001}, {"weight_decay": 0.0}, {"weight_decay": 0.0}],
                )
            ]

            self.optimizer = torch.optim.AdamW(
                lora_params, lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001
            )

        cprint(f"Task {task_idx} training...", "red", "on_yellow")

        for epoch in range(self.last_epoch + 1, args.epochs):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if args.lora_train:
                train_one_epoch(
                    self.lora_model,
                    self.criterion,
                    self.train_dataloader,
                    self.optimizer,
                    self.device,
                    epoch,
                    args.clip_max_norm,
                    ema=self.ema,
                    scaler=self.scaler,
                    task_idx=task_idx,
                    data_ratio=data_ratio,
                    pseudo_label=args.pseudo_label,
                    distill_attn=args.distill_attn,
                    teacher_path=args.teacher_path,
                    base_model=self.model,
                )

                self.lr_scheduler.step()

                if self.output_dir:
                    if (epoch + 1) % args.checkpoint_step == 0:
                        lora_pt = (
                            self.output_dir / f"lora_{data_ratio}_t{task_idx}_{epoch+1}"
                        )

                        self.lora_model.save_pretrained(lora_pt)

                evaluate(
                    self.lora_model,
                    self.criterion,
                    self.postprocessor,
                    self.val_dataloader,
                    base_ds,
                    self.device,
                )

            else:
                train_one_epoch(
                    self.model,
                    self.criterion,
                    self.train_dataloader,
                    self.optimizer,
                    self.device,
                    epoch,
                    args.clip_max_norm,
                    ema=self.ema,
                    scaler=self.scaler,
                    task_idx=task_idx,
                    data_ratio=data_ratio,
                    pseudo_label=args.pseudo_label,
                    distill_attn=args.distill_attn,
                    teacher_path=args.teacher_path,
                )

                self.lr_scheduler.step()

                module = self.ema.module if self.ema else self.model

                if self.output_dir:
                    if (epoch + 1) % args.checkpoint_step == 0:
                        checkpoint_path = (
                            self.output_dir / f"{data_ratio}_t{task_idx}_{epoch+1}e.pth"
                        )
                        dist.save_on_master(self.state_dict(epoch), checkpoint_path)

                evaluate(
                    module,
                    self.criterion,
                    self.postprocessor,
                    self.val_dataloader,
                    base_ds,
                    self.device,
                )

    def val(
        self,
    ):
        self.eval()

        args = self.cfg
        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)

        if args.lora_val:
            lora_model = PeftModel.from_pretrained(self.model, args.lora_id)
            lora_model.eval()

            evaluate(
                lora_model,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                base_ds,
                self.device,
            )

        else:
            module = self.ema.module if self.ema else self.model

            evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                base_ds,
                self.device,
            )
