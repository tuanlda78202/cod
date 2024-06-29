from src.data.cococl.mtsd_cl import MTSDDetectionCL
from src.data.transforms import Compose


def run():
    transformer = Compose([
                           {"type": "RandomPhotometricDistort", "p": 0.5},
                           {"type": "RandomZoomOut", "fill": 0},
                            {"type": "RandomIoUCrop", "p": 0.8},
                            {"type": "SanitizeBoundingBox", "min_size": 1},
                            {"type": "RandomHorizontalFlip"},
                           {"type": "Resize", "size": [640, 640]},
                            {"type": "ToImageTensor"},
                            {"type": "ConvertDtype"},
                            {"type": "SanitizeBoundingBox", "min_size": 1},
                            {"type": "ConvertBox", "out_fmt": "cxcywh", "normalize": True},
                           ] )
    cl = MTSDDetectionCL(img_folder="mtsd/mtsd_train/",
                         ann_file="mtsd/annotations/instances_train_c1.json",
                         task_idx=0,
                         data_ratio="7010s",
                         transforms=transformer,
                         return_masks=None,
                         cache_mode=None,
                         buffer_mode=None,
                         buffer_rate=None)
    # print(cl)
    ids = [11136,
     11845,
     3068,
     9380,
     1216]
    for i in ids:
        item = cl.__getitem__(i)
        print(item)

if __name__ == '__main__':
    run()
