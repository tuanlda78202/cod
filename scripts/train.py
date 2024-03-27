import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import argparse

import src.misc.dist as dist
from src.core import YAMLConfig
from src.solver import TASKS


def main(
    args,
) -> None:
    """main"""
    dist.init_distributed()

    assert not all(
        [args.tuning, args.resume]
    ), "Only support from_scratch or resume or tuning at one time"

    cfg = YAMLConfig(
        args.config, resume=args.resume, use_amp=args.amp, tuning=args.tuning
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
