def extra_epoch_for_replay(
    args,
    dataset_name: str,
    data_loader: Iterable,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    device: torch.device,
    rehearsal_classes=None,
    current_classes=None,
):
    """
    Run additional epoch to collect replay buffer.
    1. initialize prefeter, (icarl) feature extractor and prototype.d
    2. run rehearsal training.dd
    3. (icarl) detach values in rehearsal_classes.
    """
    # current_classes = [2, 3, 4]
    prefetcher = create_prefetcher(dataset_name, data_loader, device, args)

    with torch.no_grad():
        for idx in tqdm(
            range(len(data_loader)), disable=not utils.is_main_process()
        ):  # targets
            (
                samples,
                targets,
            ) = prefetcher.next()

            # Set up extra training to collect replay data
            rehearsal_classes = rehearsal_training(
                args,
                samples,
                targets,
                model,
                criterion,
                rehearsal_classes,
                current_classes,
            )

            if idx % 100 == 0:
                torch.cuda.empty_cache()

            # 정완 디버그
            if args.debug:
                if idx == args.num_debug_dataset:
                    break

    return rehearsal_classes


def construct_replay_extra_epoch(
    args, Divided_Classes, model, criterion, device, rehearsal_classes={}, task_num=0
):

    # 0. Initialization
    extra_epoch = True
    print(f"already buffer state number : {len(rehearsal_classes)}")

    # 0.1. If you are not use the construct replay method, so then you use the real task number of training step.
    if args.Construct_Replay:
        task_num = args.start_task

    # 1. Call the appropriate dataset for the current task (retrieve data related to the trained task, Task 0).
    # Retrieve all data to compose a buffer with a single GPU (more accurate).
    # list_CC : collectable class index
    dataset_train, data_loader_train, _, list_CC = Incre_Dataset(
        task_num, args, Divided_Classes, extra_epoch
    )

    # 2. Extra epoch, Measure the loss of all images
    rehearsal_classes = extra_epoch_for_replay(
        args,
        dataset_name="",
        data_loader=data_loader_train,
        model=model,
        criterion=criterion,
        device=device,
        current_classes=list_CC,
        rehearsal_classes=rehearsal_classes,
    )

    # 3. Save the collected buffer to a specific file
    if args.Rehearsal_file is None:
        args.Rehearsal_file = args.output_dir

    # Create the folder for the rehearsal file path if it does not exist
    os.makedirs(os.path.dirname(args.Rehearsal_file), exist_ok=True)
    rehearsal_classes = merge_rehearsal_process(
        args=args,
        task=task_num,
        dir=args.Rehearsal_file,
        rehearsal=rehearsal_classes,
        epoch=0,
        limit_memory_size=args.limit_image,
        gpu_counts=utils.get_world_size(),
        list_CC=list_CC,
    )

    print(colored(f"Complete constructing buffer", "red", "on_yellow"))

    return rehearsal_classes
