import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint

from src.core import register


__all__ = [
    "RTDETR",
]


@register
class RTDETR(nn.Module):
    __inject__ = [
        "backbone",
        "encoder",
        "decoder",
    ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None, task_idx=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        self.task_idx = task_idx
        
        if self.multi_scale and self.training and self.task_idx == 0:
            cprint(f"Multi-scale first task training: {self.multi_scale}", "red")

    def forward(self, x, targets=None):
        if self.multi_scale and self.training and self.task_idx == 0:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])

        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)

        return x

    def deploy(
        self,
    ):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self
