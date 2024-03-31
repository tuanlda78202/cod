import util.misc as utils
from datasets import build_dataset
from torch.utils.data import DataLoader, ConcatDataset
import torch
import numpy as np
from termcolor import colored
import copy


# * Incremental Data Loader
def CLDataLoader(task_num, incremental_classes, args):
    current_classes = incremental_classes[task_num]
    all_classes = sum(incremental_classes[: task_num + 1], [])
    is_eval_mode = args.eval

    # * Training
    if not is_eval_mode:
        print(f"Current classes for Training: {current_classes}")

        train_dataset = build_dataset(
            image_set="train", args=args, class_ids=current_classes
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=args.drop_last,
            collate_fn=utils.collate_fn,
            pin_memory=args.pin_memory,
        )

        return train_dataset, train_loader

    # * Evaluation
    else:
        target_classes = all_classes
        print(
            colored(
                f"Current classes for Evaluation: {target_classes}", "blue", "on_yellow"
            )
        )

        val_dataset = build_dataset(
            image_set="val", args=args, class_ids=target_classes
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=args.drop_last,
            collate_fn=utils.collate_fn,
            pin_memory=args.pin_memory,
        )

        return val_dataset, val_loader


# Rehearsal Dataset


def img_id_config_no_circular_training(args, re_dict):
    if args.Sampling_strategy == "icarl":
        keys = []
        for cls, val in re_dict.items():
            img_ids = np.array(val[1])
            keys.extend(list(img_ids[:, 0].astype(int)))

        no_duple_keys = list(set(keys))
        print(f"not duple keys :{len(no_duple_keys)}")
        return no_duple_keys
    else:
        return list(re_dict.keys())


class CustomDataset(torch.utils.data.Dataset):
    """
    Replay Buffer configuration
    1. Weight based Circular Experience Replay (WCER)
    2. Fisher based Circular Experience Replay (FCER)
    3. Fisher based ER
    """

    def __init__(self, args, re_dict, old_classes):
        self.re_dict = copy.deepcopy(re_dict)
        self.old_classes = old_classes

        if args.CER == "uniform" and args.AugReplay:
            self.weights = None
            self.keys = list(self.re_dict.keys())
            self.datasets = build_dataset(
                image_set="train",
                args=args,
                class_ids=self.old_classes,
                img_ids=self.keys,
            )
            self.fisher_softmax_weights = None

        else:
            self.weights = None
            self.fisher_softmax_weights = None
            self.keys = img_id_config_no_circular_training(args, re_dict)
            self.datasets = build_dataset(
                image_set="train",
                args=args,
                class_ids=self.old_classes,
                img_ids=self.keys,
            )

    def __len__(self):
        return len(self.datasets)

    def __repr__(self):
        print(f"Data key presented in buffer : {self.old_classes}")

    def __getitem__(self, idx):
        samples, targets = self.datasets[idx]

        return samples, targets


class NewDatasetSet(torch.utils.data.Dataset):
    def __init__(self, args, datasets, OldDataset, AugReplay=False, Mosaic=False):
        self.args = args
        self.Datasets = datasets  # now task
        self.Rehearsal_dataset = OldDataset
        self.AugReplay = AugReplay
        if self.AugReplay == True:
            self.old_length = (
                len(self.Rehearsal_dataset)
                if dist.get_world_size() == 1
                else int(len(self.Rehearsal_dataset) // dist.get_world_size())
            )  # 4

    def __len__(self):
        return len(self.Datasets)

    def __getitem__(self, index):
        img, target = self.Datasets[index]  # No normalize pixel, Normed Targets
        if self.AugReplay == True:
            if self.args.CER == "uniform":  # weight CER
                index = np.random.choice(np.arange(len(self.Rehearsal_dataset)))
                O_img, O_target, _, _ = self.Rehearsal_dataset[
                    index
                ]  # No shuffle because weight sorting.
                return img, target, O_img, O_target

        return img, target


def CombineDataset(
    args,
    OldData,
    CurrentDataset,
    Worker,
    old_classes,
):
    """MixReplay arguments is only used in MixReplay"""
    OldDataset = CustomDataset(args, OldData, old_classes)

    if args.AugReplay and not args.MixReplay:
        NewTaskDataset = NewDatasetSet(args, CurrentDataset, OldDataset, AugReplay=True)

    if args.Replay and not args.AugReplay and not args.MixReplay and not args.Mosaic:
        CombinedDataset = ConcatDataset([OldDataset, CurrentDataset])
        NewTaskDataset = NewDatasetSet(
            args, CombinedDataset, OldDataset, AugReplay=False
        )

    print(
        colored(
            f"Current Dataset length : {len(CurrentDataset)}\nTotal Dataset length : {len(CurrentDataset)} +  Old Dataset length : {len(OldData)}\n********** Success combined Datasets ***********",
            "blue",
        )
    )

    CombinedLoader = DataLoader(
        NewTaskDataset,
        collate_fn=utils.collate_fn,
        num_workers=Worker,
        pin_memory=True,
        prefetch_factor=args.prefetch,
    )  # worker_init_fn=worker_init_fn, persistent_workers=args.AugReplay)

    return NewTaskDataset, CombinedLoader
