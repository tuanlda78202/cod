import torch
from copy import deepcopy


def construct_replay_extra_epoch(
    args, Divided_Classes, model, criterion, device, rehearsal_classes={}, task_num=0
):

    # 0. Initialization
    extra_epoch = True
    print(f"already buffer state number : {len(rehearsal_classes)}")

    # 0.1. If you are not use the construct replay method, so then you use the real task number of training step.
    if args.Construct_Replay:
        task_num = args.start_task

    # 1. 현재 테스크에 맞는 적절한 데이터 셋 호출 (학습한 테스크, 0번 테스크에 해당하는 내용을 가져와야 함)
    #    하나의 GPU로 Buffer 구성하기 위해서(더 정확함) 모든 데이터 호출
    # list_CC : collectable class index
    dataset_train, data_loader_train, _, list_CC = Incre_Dataset(
        task_num, args, Divided_Classes, extra_epoch
    )

    # 2. Extra epoch, 모든 이미지들의 Loss를 측정
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

    # 3. 수집된 Buffer를 특정 파일에 저장
    if args.Rehearsal_file is None:
        args.Rehearsal_file = args.output_dir
    # Rehearsal_file 경로의 폴더가 없을 경우 생성
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


def extra_epoch_for_replay(
    args,
    dataset_name: str,
    data_loader: Iterable,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    device: torch.device,
    rehearsal_classes,
    current_classes,
):
    """
    Run additional epoch to collect replay buffer.
    1. initialize prefeter, (icarl) feature extractor and prototype.
    2. run rehearsal training.
    3. (icarl) detach values in rehearsal_classes.
    """
    # current_classes = [2, 3, 4]
    prefetcher = create_prefetcher(dataset_name, data_loader, device, args)

    fe = icarl_feature_extractor_setup(args, model)
    proto = icarl_prototype_setup(args, fe, device, current_classes)

    with torch.no_grad():
        for idx in tqdm(
            range(len(data_loader)), disable=not utils.is_main_process()
        ):  # targets
            samples, targets, _, _ = prefetcher.next()

            # extra training을 통해서 replay 데이터를 수집하도록 설정
            rehearsal_classes = icarl_rehearsal_training(
                args,
                samples,
                targets,
                fe,
                proto,
                device,
                rehearsal_classes,
                current_classes,
            )
            if idx % 100 == 0:
                torch.cuda.empty_cache()

            # 정완 디버그
            if args.debug:
                if idx == args.num_debug_dataset:
                    break

        # * rehearsal_classes : [feature_sum, [[image_ids, difference] ...]]
        for key, val in rehearsal_classes.items():
            val[0] = val[0].detach().cpu()

    return rehearsal_classes


@torch.no_grad()
def icarl_feature_extractor_setup(args, model):
    """
    In iCaRL, buffer manager collect samples closed to the mean of features of corresponding class.
    This function set up feature extractor for collecting.
    """
    if args.distributed:
        feature_extractor = deepcopy(model.module.backbone)
    else:
        feature_extractor = deepcopy(
            model.backbone
        )  # distributed:model.module.backbone

    for n, p in feature_extractor.named_parameters():
        p.requires_grad = False

    return feature_extractor


@torch.no_grad()
def icarl_prototype_setup(args, feature_extractor, device, current_classes):
    """
    In iCaRL, buffer manager collect samples closed to the mean of features of corresponding class.
    This function set up prototype-mean of features of corresponding class-.
    Prototype can be the 'criteria' to select closest samples.
    """

    feature_extractor.eval()
    proto = defaultdict(int)

    for cls in current_classes:
        _dataset, _data_loader, _sampler = IcarlDataset(args=args, single_class=cls)
        if _dataset == None:
            continue

        _cnt = 0
        for samples, targets, _, _ in tqdm(
            _data_loader,
            desc=f"Prototype:class_{cls}",
            disable=not utils.is_main_process(),
        ):
            samples = samples.to(device)
            feature, _ = feature_extractor(samples)
            feature_0 = feature[0].tensors
            proto[cls] += feature_0
            _cnt += 1
            if args.debug and _cnt == 10:
                break

        try:
            proto[cls] = proto[cls] / _dataset.__len__()
        except ZeroDivisionError:
            pass
        if args.debug and cls == 10:
            break

    return proto


@torch.no_grad()
def icarl_rehearsal_training(
    args,
    samples,
    targets,
    fe: torch.nn.Module,
    proto: Dict,
    device: torch.device,
    rehearsal_classes,
    current_classes,
):
    """
    iCaRL buffer collection.

    rehearsal_classes : [feature_sum, [[image_ids, difference] ...]]
    """

    fe.eval()
    samples.to(device)

    feature, pos = fe(samples)
    feat_tensor = feature[0].tensors  # TODO: cpu or cuda?

    for bt_idx in range(feat_tensor.shape[0]):
        feat_0 = feat_tensor[bt_idx]
        target = targets[bt_idx]
        label_tensor = targets[bt_idx]["labels"]
        label_tensor_unique = torch.unique(label_tensor)
        label_list_unique = label_tensor_unique.tolist()

        for label in label_list_unique:
            try:
                class_mean = proto[label]
            except KeyError:
                print(f"label: {label} don't in prototype: {proto.keys()}")
                continue
            try:
                if label in rehearsal_classes:  # rehearsal_classes[label] exist
                    rehearsal_classes[label][0] = rehearsal_classes[label][0].to(device)

                    exemplar_mean = (rehearsal_classes[label][0] + feat_0) / (
                        len(rehearsal_classes[label]) + 1
                    )
                    difference = torch.mean(
                        torch.sqrt(torch.sum((class_mean - exemplar_mean) ** 2, axis=1))
                    ).item()

                    rehearsal_classes[label][0] = rehearsal_classes[label][0]
                    rehearsal_classes[label][0] += feat_0
                    rehearsal_classes[label][1].append(
                        [target["image_id"].item(), difference]
                    )

                else:
                    # "initioalization"
                    difference = torch.argmin(
                        torch.sqrt(torch.sum((class_mean - feat_0) ** 2, axis=0))
                    ).item()  # argmin is true????
                    rehearsal_classes[label] = [
                        feat_0,
                        [
                            [target["image_id"].item(), difference],
                        ],
                    ]
            except Exception as e:
                print(f"Error opening image: {e}")
                difference = torch.argmin(
                    torch.sqrt(torch.sum((class_mean - feat_0) ** 2, axis=0))
                ).item()  # argmin is true????
                rehearsal_classes[label] = [
                    feat_0,
                    [
                        [target["image_id"].item(), difference],
                    ],
                ]

            rehearsal_classes[label][1].sort(key=lambda x: x[1])  # sort with difference

    # construct rehearsal (3) - reduce exemplar set
    # for label, data in tqdm(rehearsal_classes.items(), desc='Reduce_exemplar:', disable=not utils.is_main_process()):
    #     try:
    #         data[1] = data[1][:args.limit_image]
    #     except:
    #         continue

    return rehearsal_classes
