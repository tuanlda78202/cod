import numpy as np


# * Divide classes
def divide_classes_randomly(total_classes, ratios, seed=123):
    np.random.seed(seed)
    np.random.shuffle(total_classes)
    divided_classes = []
    start_idx = 0

    for ratio in ratios:
        end_idx = start_idx + ratio
        divided_classes.append(total_classes[start_idx:end_idx])
        start_idx = end_idx

    return divided_classes


def data_setting(ratio: str, random_setting: bool = False):
    flatten_list = lambda nested_list: [
        item for sublist in nested_list for item in sublist
    ]
    total_classes = list(range(1, 91))
    divided_classes = [
        list(range(1, 46)),  # 45 classes
        list(range(46, 56)),  # 10 classes
        list(range(56, 66)),  # 10 classes
        list(range(66, 80)),  # 14 classes
        list(range(80, 91)),  # 11 classes
    ]

    ratio_to_classes = {
        "4040": [divided_classes[0], flatten_list(divided_classes[1:])],
        "402020": [
            divided_classes[0],
            flatten_list(divided_classes[1:3]),
            flatten_list(divided_classes[3:]),
        ],
        "4010101010": divided_classes,
        "7010": [flatten_list(divided_classes[:-1]), divided_classes[-1]],
        "80": [total_classes, total_classes],
        "1010": [list(range(1, 11)), list(range(11, 22))],
        "20": [list(range(1, 22))],
    }

    divided_classes_detail = ratio_to_classes.get(ratio, total_classes)

    # * Various order testing in CL-DETR
    if random_setting:
        ratios = [40, 10, 10, 10, 10]

        divided_classes_detail = divide_classes_randomly(total_classes, ratios)

    return divided_classes_detail
