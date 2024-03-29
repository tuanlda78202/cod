import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import argparse

import src.misc.dist as dist
from src.core import YAMLConfig
from src.solver import TASKS

import wandb
import torch
import numpy as np


def set_seed_and_config():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def main(
    args,
) -> None:
    """main"""
    set_seed_and_config()
    dist.init_distributed()

    assert not all(
        [args.tuning, args.resume]
    ), "Only support from_scratch or resume or tuning at one time"

    cfg = YAMLConfig(
        args.config, resume=args.resume, use_amp=args.amp, tuning=args.tuning
    )

    wandb_mode = True
    if wandb_mode:
        wandb.init(
            project="cod", entity="tuanlda78202", name="4040_l40_2e_fq_disattn_buffer10"
        )
        solver = TASKS[cfg.yaml_cfg["task"]](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        "-c",
        default="configs/rtdetr/rtdetr_r50vd_6x_coco.yml",
        type=str,
    )
    parser.add_argument(
        "--resume",
        "-r",
        type=str,
    )
    parser.add_argument(
        "--tuning",
        "-t",
        default="",
        type=str,
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    main(args)
