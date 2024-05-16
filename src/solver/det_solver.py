from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate

from termcolor import cprint


class DetSolver(BaseSolver):

    def fit(
        self,
    ):
        self.train()

        args = self.cfg

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        task_idx = self.train_dataloader.dataset.task_idx
        data_ratio = self.train_dataloader.dataset.data_ratio

        train_text_feat = self.train_dataloader.dataset.text_feat
        val_text_feat = self.val_dataloader.dataset.text_feat

        # self.model = torch.compile(self.model)

        cprint(f"Task {task_idx} training...", "red", "on_yellow")

        for epoch in range(self.last_epoch + 1, args.epochs):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

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
                text_feat=train_text_feat,
            )

            self.lr_scheduler.step()

            module = self.ema.module if self.ema else self.model

            if self.output_dir:
                if (epoch + 1) % args.checkpoint_step == 0:
                    checkpoint_path = (
                        self.output_dir / f"{data_ratio}_t{task_idx}_{epoch+1}e.pth"
                    )
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path)

            ap = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                base_ds,
                self.device,
                text_feat=val_text_feat,
            )

    def val(
        self,
    ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        text_feat = self.val_dataloader.dataset.text_feat

        module = self.ema.module if self.ema else self.model

        evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            base_ds,
            self.device,
            text_feat=text_feat,
        )
