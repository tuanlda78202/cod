"""
Add utility of caching images on memory
"""

from torchvision.datasets import CocoDetection as CCD
from PIL import Image
import os
import os.path
import tqdm
from io import BytesIO
from termcolor import colored, cprint


class CocoCache(CCD):
    """
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root,
        annFile,
        transform=None,
        target_transform=None,
        transforms=None,
        cache_mode=False,
        ids_list=None,
        class_ids=None,
        buffer_ids=None,
        buffer_rate=None,
        buffer_mode=None,
    ):
        super(CocoCache, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)

        self.ids = list(sorted(self.coco.imgs.keys())) if ids_list == None else ids_list

        if not isinstance(class_ids, list):
            class_ids = list(class_ids)
        self.class_ids = class_ids

        if class_ids is not None and ids_list == None:
            self.ids = []

            for c_idx in self.class_ids:
                img_ids = self.coco.getImgIds(catIds=c_idx)
                self.ids.extend(img_ids)

            cprint(
                f"Original Images: {len(set(self.ids))}",
                "green",
                "on_red",
            )

            if buffer_mode:
                total_buffer = 0
                self.buffer_ids = buffer_ids

                for b_idx in self.buffer_ids:
                    buffer_ids = self.coco.getImgIds(catIds=b_idx)
                    buffer_size = int(len(buffer_ids) * buffer_rate)
                    total_buffer += buffer_size
                    self.ids.extend(buffer_ids[:buffer_size])
                cprint(
                    f"Buffer Images: {total_buffer}\n{len(self.buffer_ids)} Buffer Classes: {self.buffer_ids}",
                    "green",
                    "on_red",
                )
                print("---------------------------------")

            self.ids = list(set(self.ids))

        self.cache_mode = cache_mode

        if cache_mode:
            self.cache = {}
            self.cache_images()

        cprint(
            f"Total Images: {len(self.ids)}\n{len(self.class_ids)} Task Classes: {self.class_ids}",
            "red",
            "on_cyan",
        )

    def cache_images(self):
        self.cache = {}
        for index, img_id in zip(tqdm.trange(len(self.ids)), self.ids):
            path = self.coco.loadImgs(img_id)[0]["file_name"]

            with open(os.path.join(self.root, path), "rb") as f:
                self.cache[path] = f.read()

    def get_image(self, path):
        if self.cache_mode:
            if path not in self.cache.keys():
                with open(os.path.join(self.root, path), "rb") as f:
                    self.cache[path] = f.read()
            return Image.open(BytesIO(self.cache[path])).convert("RGB")

        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        # TODO : 기존에는 모든 존재하는 이미지 ID를 가져와서 사용하던 것을 변환 -> 미리 선언한 Class에 해당하는 이미지들만 선별

        if (
            self.class_ids is not None
        ):  # 클래스 IDS에 해당하는 Target만 가져오는 것이 핵심
            target = [
                value
                for value in coco.loadAnns(coco.getAnnIds(img_id))
                if value["category_id"] in self.class_ids
            ]
        else:
            ann_ids = coco.getAnnIds(imgIds=img_id)
            target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(int(img_id))[0]["file_name"]
        img = self.get_image(path)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)
