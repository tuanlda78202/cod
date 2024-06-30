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
    print("Ratio: ", ratio)
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
    divided_classes_mtsd_s1 = [
        [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129],
        [130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221]]
    deviced_classes_mtsd_s2 = [[],[]]
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
        "7010s": [divided_classes_mtsd_s1[0][:70], divided_classes_mtsd_s1[1][:10]],
        "4040s": [divided_classes_mtsd_s1[0][:40], divided_classes_mtsd_s1[1][:40]],

    }

    divided_classes_detail = ratio_to_classes.get(ratio, total_classes)
    print("Detail: ", divided_classes_detail)
    # * Various order testing in CL-DETR
    if random_setting:
        ratios = [40, 10, 10, 10, 10]

        divided_classes_detail = divide_classes_randomly(total_classes, ratios)

    return divided_classes_detail
