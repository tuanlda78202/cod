import torch
import torch.nn as nn

import re
import copy

from .config import BaseConfig
from .yaml_utils import load_config, merge_config, create, merge_dict


class YAMLConfig(BaseConfig):
    def __init__(self, cfg_path: str, **kwargs) -> None:
        super().__init__()

        cfg = load_config(cfg_path)
        merge_dict(cfg, kwargs)

        self.yaml_cfg = cfg

        self.log_step = cfg.get("log_step", 100)
        self.checkpoint_step = cfg.get("checkpoint_step", 1)
        self.epochs = cfg.get("epochs", -1)
        self.resume = cfg.get("resume", "")
        self.tuning = cfg.get("tuning", "")
        self.sync_bn = cfg.get("sync_bn", False)
        self.output_dir = cfg.get("output_dir", None)

        self.use_ema = cfg.get("use_ema", False)
        self.use_amp = cfg.get("use_amp", False)
        self.autocast = cfg.get("autocast", dict())
        self.find_unused_parameters = cfg.get("find_unused_parameters", None)
        self.clip_max_norm = cfg.get("clip_max_norm", 0.0)

        # *  CL General
        self.start_task = cfg.get("start_task", None)
        self.total_tasks = cfg.get("total_tasks", None)

        # * CL Strategy
        self.pseudo_label = cfg.get("pseudo_label", None)
        self.distill_attn = cfg.get("distill_attn", None)
        self.teacher_path = cfg.get("teacher_path", None)

        # * CL Eval
        self.fpp: bool = False

        # *  WandB
        self.wandb_name = cfg.get("wandb_name", None)
        self.wandb_project = cfg.get("wandb_project", "cod")
        self.wandb_entity = cfg.get("wandb_entity", "tuanlda78202")
        self.config_info = copy.deepcopy(self.yaml_cfg)

        # * CL Rehearsal
        self.rehearsal = cfg.get("rehearsal", False)
        self.construct_replay: bool = False
        self.sampling_strategy: str = None
        self.sampling_mode: str = None
        self.least_sample: int = None
        self.limit_sample: int = None

        self.augment_replay: bool = False
        self.mix_replay: bool = False
        self.mosaic: bool = False

    @property
    def model(
        self,
    ) -> torch.nn.Module:
        if self._model is None and "model" in self.yaml_cfg:
            merge_config(self.yaml_cfg)
            self._model = create(self.yaml_cfg["model"])
        return self._model

    @property
    def postprocessor(
        self,
    ) -> torch.nn.Module:
        if self._postprocessor is None and "postprocessor" in self.yaml_cfg:
            merge_config(self.yaml_cfg)
            self._postprocessor = create(self.yaml_cfg["postprocessor"])
        return self._postprocessor

    @property
    def criterion(
        self,
    ):
        if self._criterion is None and "criterion" in self.yaml_cfg:
            merge_config(self.yaml_cfg)
            self._criterion = create(self.yaml_cfg["criterion"])
        return self._criterion

    @property
    def optimizer(
        self,
    ):
        if self._optimizer is None and "optimizer" in self.yaml_cfg:
            merge_config(self.yaml_cfg)
            params = self.get_optim_params(self.yaml_cfg["optimizer"], self.model)
            self._optimizer = create("optimizer", params=params)

        return self._optimizer

    @property
    def lr_scheduler(
        self,
    ):
        if self._lr_scheduler is None and "lr_scheduler" in self.yaml_cfg:
            merge_config(self.yaml_cfg)
            self._lr_scheduler = create("lr_scheduler", optimizer=self.optimizer)
            print("Initial lr: ", self._lr_scheduler.get_last_lr())

        return self._lr_scheduler

    @property
    def train_dataloader(self, task_idx: int = 1):
        if self._train_dataloader is None and "train_dataloader" in self.yaml_cfg:
            merge_config(self.yaml_cfg)
            self._train_dataloader = create("train_dataloader")
            self._train_dataloader.shuffle = self.yaml_cfg["train_dataloader"].get(
                "shuffle", False
            )

        return self._train_dataloader

    @property
    def val_dataloader(
        self,
    ):
        if self._val_dataloader is None and "val_dataloader" in self.yaml_cfg:
            merge_config(self.yaml_cfg)
            self._val_dataloader = create("val_dataloader")
            self._val_dataloader.shuffle = self.yaml_cfg["val_dataloader"].get(
                "shuffle", False
            )

        return self._val_dataloader

    @property
    def ema(
        self,
    ):
        if self._ema is None and self.yaml_cfg.get("use_ema", False):
            merge_config(self.yaml_cfg)
            self._ema = create("ema", model=self.model)

        return self._ema

    @property
    def scaler(
        self,
    ):
        if self._scaler is None and self.yaml_cfg.get("use_amp", False):
            merge_config(self.yaml_cfg)
            self._scaler = create("scaler")

        return self._scaler

    @staticmethod
    def get_optim_params(cfg: dict, model: nn.Module):
        """
        E.g.:
            ^(?=.*a)(?=.*b).*$         means including a and b
            ^((?!b.)*a((?!b).)*$       means including a but not b
            ^((?!b|c).)*a((?!b|c).)*$  means including a but not (b | c)
        """
        assert "type" in cfg, ""
        cfg = copy.deepcopy(cfg)

        if "params" not in cfg:
            return model.parameters()

        assert isinstance(cfg["params"], list), ""

        param_groups = []
        visited = []
        for pg in cfg["params"]:
            pattern = pg["params"]
            params = {
                k: v
                for k, v in model.named_parameters()
                if v.requires_grad and len(re.findall(pattern, k)) > 0
            }
            pg["params"] = params.values()
            param_groups.append(pg)
            visited.extend(list(params.keys()))

        names = [k for k, v in model.named_parameters() if v.requires_grad]

        if len(visited) < len(names):
            unseen = set(names) - set(visited)
            params = {
                k: v
                for k, v in model.named_parameters()
                if v.requires_grad and k in unseen
            }
            param_groups.append({"params": params.values()})
            visited.extend(list(params.keys()))

        assert len(visited) == len(names), ""

        return param_groups
