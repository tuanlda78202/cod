def IcarlDataset(args, single_class: int):
    """
    For initiating prototype-mean of the feature of corresponding, single class-, dataset composed to single class is needed.
    """
    dataset = build_dataset(image_set="extra", args=args, class_ids=[single_class])
    if len(dataset) == 0:
        return None, None, None

    if args.distributed:
        if args.cache_mode:
            sampler = samplers.NodeDistributedSampler(dataset)
        else:
            sampler = samplers.DistributedSampler(dataset)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    batch_sampler = torch.utils.data.BatchSampler(
        sampler, args.batch_size, drop_last=True
    )
    data_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return dataset, data_loader, sampler
