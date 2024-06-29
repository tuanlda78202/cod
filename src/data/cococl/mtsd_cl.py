import torch
import torch.utils.data

import torchvision

torchvision.disable_beta_transforms_warning()

from torchvision import datapoints
from pycocotools import mask as coco_mask

from src.core import register
from .coco_cache import CocoCache
from .cl_utils import data_setting

__all__ = ["MTSDDetectionCL"]


@register
class MTSDDetectionCL(CocoCache):
    __inject__ = ["transforms"]
    __share__ = ["remap_mscoco_category"]

    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        return_masks,
        cache_mode,
        task_idx,
        data_ratio,
        buffer_mode,
        buffer_rate,
        remap_mscoco_category=False,
        img_ids=None,
    ):
        self.task_idx = task_idx
        self.data_ratio = data_ratio
        divided_classes = data_setting(data_ratio)
        class_ids_current = divided_classes[self.task_idx]
        buffer_ids = list(set(list(range(1, 91))) - set(class_ids_current))

        super().__init__(
            img_folder,
            ann_file,
            class_ids=class_ids_current,
            buffer_ids=buffer_ids,
            cache_mode=cache_mode,
            ids_list=img_ids,
            buffer_rate=buffer_rate,
            buffer_mode=buffer_mode,
        )

        cats = {}
        for class_id in class_ids_current:
            try:
                cats[class_id] = self.coco.cats[class_id]
            except KeyError:
                pass
        self.coco.cats = cats

        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, remap_mscoco_category)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __getitem__(self, idx):
        img, target = super(MTSDDetectionCL, self).__getitem__(idx)

        image_id = self.ids[idx]
        # print("Item: ", idx, image_id)
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)

        # ['boxes', 'masks', 'labels']:
        if "boxes" in target:
            target["boxes"] = datapoints.BoundingBox(
                target["boxes"],
                format=datapoints.BoundingBoxFormat.XYXY,
                spatial_size=img.size[::-1],
            )

        if "masks" in target:
            target["masks"] = datapoints.Mask(target["masks"])

        if self._transforms is not None:
            try:
                img, target = self._transforms(img, target)
            except Exception as e:
                print(e)
                print("Failed to transform image and target")
                print(idx, image_id,target)
                exit(-1)

        return img, target

    def extra_repr(self) -> str:
        s = f" img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n"
        s += f" return_masks: {self.return_masks}\n"
        if hasattr(self, "_transforms") and self._transforms is not None:
            s += f" transforms:\n   {repr(self._transforms)}"
        return s


def convert_coco_poly_to_mask(segmentation, height, width):
    masks = []
    for polygons in segmentation:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, remap_mscoco_category=False):
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        if self.remap_mscoco_category:
            classes = [mtsd_category2label[obj["category_id"]] for obj in anno]
        else:
            classes = [obj["category_id"] for obj in anno]

        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor(
            [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno]
        )
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        target["size"] = torch.as_tensor([int(w), int(h)])

        return image, target


mtsd_category2name = {1: "regulatory--no-stopping--g15",
     2: "complementary--accident-area--g3",
     3: "complementary--both-directions--g1",
     4: "complementary--buses--g1",
     5: "complementary--chevron-left--g1",
     6: "complementary--chevron-left--g2",
     7: "complementary--chevron-left--g3",
     8: "complementary--chevron-left--g4",
     9: "complementary--chevron-left--g5",
     10: "complementary--chevron-right--g1",
     11: "complementary--chevron-right--g3",
     12: "complementary--chevron-right--g4",
     13: "complementary--chevron-right--g5",
     14: "complementary--chevron-right-unsure--g6",
     15: "complementary--distance--g1",
     16: "complementary--distance--g2",
     17: "complementary--distance--g3",
     18: "complementary--except-bicycles--g1",
     19: "complementary--extent-of-prohibition-area-both-direction--g1",
     20: "complementary--go-left--g1",
     21: "complementary--go-right--g1",
     22: "complementary--go-right--g2",
     23: "complementary--keep-left--g1",
     24: "complementary--keep-right--g1",
     25: "complementary--maximum-speed-limit-15--g1",
     26: "complementary--maximum-speed-limit-20--g1",
     27: "complementary--maximum-speed-limit-25--g1",
     28: "complementary--maximum-speed-limit-30--g1",
     29: "complementary--maximum-speed-limit-35--g1",
     30: "complementary--maximum-speed-limit-40--g1",
     31: "complementary--maximum-speed-limit-45--g1",
     32: "complementary--maximum-speed-limit-50--g1",
     33: "complementary--maximum-speed-limit-55--g1",
     34: "complementary--maximum-speed-limit-70--g1",
     35: "complementary--maximum-speed-limit-75--g1",
     36: "complementary--obstacle-delineator--g1",
     37: "complementary--obstacle-delineator--g2",
     38: "complementary--one-direction-left--g1",
     39: "complementary--one-direction-right--g1",
     40: "complementary--pass-right--g1",
     41: "complementary--priority-route-at-intersection--g1",
     42: "complementary--tow-away-zone--g1",
     43: "complementary--trucks--g1",
     44: "complementary--trucks-turn-right--g1",
     45: "complementary--turn-left--g2",
     46: "complementary--turn-right--g2",
     47: "information--airport--g1",
     48: "information--airport--g2",
     49: "information--bike-route--g1",
     50: "information--bus-stop--g1",
     51: "information--camp--g1",
     52: "information--central-lane--g1",
     53: "information--dead-end--g1",
     54: "information--dead-end-except-bicycles--g1",
     55: "information--disabled-persons--g1",
     56: "information--emergency-facility--g2",
     57: "information--end-of-built-up-area--g1",
     58: "information--end-of-limited-access-road--g1",
     59: "information--end-of-living-street--g1",
     60: "information--end-of-motorway--g1",
     61: "information--end-of-pedestrians-only--g2",
     62: "information--food--g2",
     63: "information--gas-station--g1",
     64: "information--highway-exit--g1",
     65: "information--hospital--g1",
     66: "information--interstate-route--g1",
     67: "information--limited-access-road--g1",
     68: "information--living-street--g1",
     69: "information--lodging--g1",
     70: "information--minimum-speed-40--g1",
     71: "information--motorway--g1",
     72: "information--no-parking--g3",
     73: "information--parking--g1",
     74: "information--parking--g2",
     75: "information--parking--g3",
     76: "information--parking--g45",
     77: "information--parking--g5",
     78: "information--parking--g6",
     79: "information--pedestrians-crossing--g1",
     80: "information--pedestrians-crossing--g2",
     81: "information--road-bump--g1",
     82: "information--safety-area--g2",
     83: "information--stairs--g1",
     84: "information--telephone--g1",
     85: "information--telephone--g2",
     86: "information--trailer-camping--g1",
     87: "regulatory--bicycles-only--g1",
     88: "regulatory--bicycles-only--g2",
     89: "regulatory--bicycles-only--g3",
     90: "regulatory--buses-only--g1",
     91: "regulatory--detour-left--g1",
     92: "regulatory--do-not-block-intersection--g1",
     93: "regulatory--do-not-stop-on-tracks--g1",
     94: "regulatory--dual-lanes-go-straight-on-left--g1",
     95: "regulatory--dual-lanes-go-straight-on-right--g1",
     96: "regulatory--dual-lanes-turn-left-no-u-turn--g1",
     97: "regulatory--dual-lanes-turn-left-or-straight--g1",
     98: "regulatory--dual-lanes-turn-right-or-straight--g1",
     99: "regulatory--dual-path-bicycles-and-pedestrians--g1",
     100: "regulatory--dual-path-pedestrians-and-bicycles--g1",
     101: "regulatory--end-of-bicycles-only--g1",
     102: "regulatory--end-of-buses-only--g1",
     103: "regulatory--end-of-maximum-speed-limit-30--g2",
     104: "regulatory--end-of-maximum-speed-limit-70--g1",
     105: "regulatory--end-of-maximum-speed-limit-70--g2",
     106: "regulatory--end-of-no-parking--g1",
     107: "regulatory--end-of-priority-road--g1",
     108: "regulatory--end-of-prohibition--g1",
     109: "regulatory--end-of-speed-limit-zone--g1",
     110: "regulatory--give-way-to-oncoming-traffic--g1",
     111: "regulatory--go-straight--g1",
     112: "regulatory--go-straight--g3",
     113: "regulatory--go-straight-or-turn-left--g1",
     114: "regulatory--go-straight-or-turn-left--g2",
     115: "regulatory--go-straight-or-turn-left--g3",
     116: "regulatory--go-straight-or-turn-right--g1",
     117: "regulatory--go-straight-or-turn-right--g3",
     118: "regulatory--height-limit--g1",
     119: "regulatory--keep-left--g1",
     120: "regulatory--keep-left--g2",
     121: "regulatory--keep-right--g1",
     122: "regulatory--keep-right--g2",
     123: "regulatory--keep-right--g4",
     124: "regulatory--keep-right--g6",
     125: "regulatory--lane-control--g1",
     126: "regulatory--left-turn-yield-on-green--g1",
     127: "regulatory--maximum-speed-limit-10--g1",
     128: "regulatory--maximum-speed-limit-100--g1",
     129: "regulatory--maximum-speed-limit-100--g3",
     130: "regulatory--maximum-speed-limit-110--g1",
     131: "regulatory--maximum-speed-limit-120--g1",
     132: "regulatory--maximum-speed-limit-15--g1",
     133: "regulatory--maximum-speed-limit-20--g1",
     134: "regulatory--maximum-speed-limit-25--g1",
     135: "regulatory--maximum-speed-limit-25--g2",
     136: "regulatory--maximum-speed-limit-30--g1",
     137: "regulatory--maximum-speed-limit-30--g3",
     138: "regulatory--maximum-speed-limit-35--g2",
     139: "regulatory--maximum-speed-limit-40--g1",
     140: "regulatory--maximum-speed-limit-40--g3",
     141: "regulatory--maximum-speed-limit-40--g6",
     142: "regulatory--maximum-speed-limit-45--g1",
     143: "regulatory--maximum-speed-limit-45--g3",
     144: "regulatory--maximum-speed-limit-5--g1",
     145: "regulatory--maximum-speed-limit-50--g1",
     146: "regulatory--maximum-speed-limit-50--g6",
     147: "regulatory--maximum-speed-limit-55--g2",
     148: "regulatory--maximum-speed-limit-60--g1",
     149: "regulatory--maximum-speed-limit-65--g2",
     150: "regulatory--maximum-speed-limit-70--g1",
     151: "regulatory--maximum-speed-limit-80--g1",
     152: "regulatory--maximum-speed-limit-90--g1",
     153: "regulatory--maximum-speed-limit-led-100--g1",
     154: "regulatory--maximum-speed-limit-led-60--g1",
     155: "regulatory--maximum-speed-limit-led-80--g1",
     156: "regulatory--minimum-safe-distance--g1",
     157: "regulatory--mopeds-and-bicycles-only--g1",
     158: "regulatory--no-bicycles--g1",
     159: "regulatory--no-bicycles--g2",
     160: "regulatory--no-bicycles--g3",
     161: "regulatory--no-buses--g3",
     162: "regulatory--no-entry--g1",
     163: "regulatory--no-hawkers--g1",
     164: "regulatory--no-heavy-goods-vehicles--g1",
     165: "regulatory--no-heavy-goods-vehicles--g2",
     166: "regulatory--no-heavy-goods-vehicles--g4",
     167: "regulatory--no-heavy-goods-vehicles--g5",
     168: "regulatory--no-heavy-goods-vehicles-or-buses--g1",
     169: "regulatory--no-left-turn--g1",
     170: "regulatory--no-left-turn--g2",
     171: "regulatory--no-left-turn--g3",
     172: "regulatory--no-mopeds-or-bicycles--g1",
     173: "regulatory--no-motor-vehicle-trailers--g1",
     174: "regulatory--no-motor-vehicles--g1",
     175: "regulatory--no-motor-vehicles--g4",
     176: "regulatory--no-motor-vehicles-except-motorcycles--g1",
     177: "regulatory--no-motor-vehicles-except-motorcycles--g2",
     178: "regulatory--no-motorcycles--g1",
     179: "regulatory--no-motorcycles--g2",
     180: "regulatory--no-overtaking--g1",
     181: "regulatory--no-overtaking--g2",
     182: "regulatory--no-overtaking--g4",
     183: "regulatory--no-overtaking--g5",
     184: "regulatory--no-overtaking-by-heavy-goods-vehicles--g1",
     185: "regulatory--no-parking--g1",
     186: "regulatory--no-parking--g2",
     187: "regulatory--no-parking--g5",
     188: "regulatory--no-parking-or-no-stopping--g1",
     189: "regulatory--no-parking-or-no-stopping--g2",
     190: "regulatory--no-parking-or-no-stopping--g3",
     191: "regulatory--no-pedestrians--g1",
     192: "regulatory--no-pedestrians--g2",
     193: "regulatory--no-pedestrians--g3",
     194: "regulatory--no-pedestrians-or-bicycles--g1",
     195: "regulatory--no-right-turn--g1",
     196: "regulatory--no-right-turn--g2",
     197: "regulatory--no-right-turn--g3",
     198: "regulatory--no-stopping--g2",
     199: "regulatory--no-stopping--g4",
     200: "regulatory--no-stopping--g5",
     201: "regulatory--no-stopping--g8",
     202: "regulatory--no-straight-through--g1",
     203: "regulatory--no-straight-through--g2",
     204: "regulatory--no-turn-on-red--g1",
     205: "regulatory--no-turn-on-red--g2",
     206: "regulatory--no-turn-on-red--g3",
     207: "regulatory--no-turns--g1",
     208: "regulatory--no-u-turn--g1",
     209: "regulatory--no-u-turn--g2",
     210: "regulatory--no-u-turn--g3",
     211: "regulatory--no-vehicles-carrying-dangerous-goods--g1",
     212: "regulatory--one-way-left--g1",
     213: "regulatory--one-way-left--g2",
     214: "regulatory--one-way-left--g3",
     215: "regulatory--one-way-right--g1",
     216: "regulatory--one-way-right--g2",
     217: "regulatory--one-way-right--g3",
     218: "regulatory--one-way-straight--g1",
     219: "regulatory--parking-restrictions--g2",
     220: "regulatory--pass-on-either-side--g1",
     221: "regulatory--pass-on-either-side--g2",
     222: "regulatory--passing-lane-ahead--g1",
     223: "regulatory--pedestrians-only--g1",
     224: "regulatory--pedestrians-only--g2",
     225: "regulatory--priority-over-oncoming-vehicles--g1",
     226: "regulatory--priority-road--g4",
     227: "regulatory--radar-enforced--g1",
     228: "regulatory--reversible-lanes--g2",
     229: "regulatory--road-closed--g1",
     230: "regulatory--road-closed--g2",
     231: "regulatory--road-closed-to-vehicles--g1",
     232: "regulatory--road-closed-to-vehicles--g3",
     233: "regulatory--roundabout--g1",
     234: "regulatory--roundabout--g2",
     235: "regulatory--shared-path-bicycles-and-pedestrians--g1",
     236: "regulatory--shared-path-pedestrians-and-bicycles--g1",
     237: "regulatory--stop--g1",
     238: "regulatory--stop--g10",
     239: "regulatory--stop--g2",
     240: "regulatory--text-four-lines",
     241: "regulatory--truck-speed-limit-60",
     242: "regulatory--weight-limit-with-trucks",
     243: "regulatory--width-limit",
     244: "warning--accidental-area-unsure",
     245: "warning--bus-stop-ahead",
     246: "warning--dip",
     247: "warning--equestrians-crossing",
     248: "warning--hairpin-curve-left",
     249: "warning--loop-270-degree",
     250: "warning--offset-roads",
     251: "warning--pedestrian-stumble-train",
     252: "warning--playground",
     253: "warning--restricted-zone",
     254: "warning--shared-lane-motorcycles-bicycles",
     255: "warning--slippery-motorcycles",
     256: "warning--trams-crossing",
     257: "warning--uneven-roads-ahead",
     258: "warning--wombat-crossing",
     259: "regulatory--width-limit--g1",
     260: "regulatory--wrong-way--g1",
     261: "regulatory--yield--g1",
     262: "warning--accidental-area-unsure--g2",
     263: "warning--added-lane-right--g1",
     264: "warning--bicycles-crossing--g1",
     265: "warning--bicycles-crossing--g2",
     266: "warning--bicycles-crossing--g3",
     267: "warning--bus-stop-ahead--g3",
     268: "warning--children--g1",
     269: "warning--children--g2",
     270: "warning--crossroads--g1",
     271: "warning--crossroads--g3",
     272: "warning--crossroads-with-priority-to-the-right--g1",
     273: "warning--curve-left--g1",
     274: "warning--curve-left--g2",
     275: "warning--curve-right--g1",
     276: "warning--curve-right--g2",
     277: "warning--dip--g2",
     278: "warning--divided-highway-ends--g1",
     279: "warning--divided-highway-ends--g2",
     280: "warning--domestic-animals--g1",
     281: "warning--domestic-animals--g3",
     282: "warning--double-curve-first-left--g1",
     283: "warning--double-curve-first-left--g2",
     284: "warning--double-curve-first-right--g1",
     285: "warning--double-curve-first-right--g2",
     286: "warning--double-reverse-curve-right--g1",
     287: "warning--double-turn-first-right--g1",
     288: "warning--dual-lanes-right-turn-or-go-straight--g1",
     289: "warning--emergency-vehicles--g1",
     290: "warning--equestrians-crossing--g2",
     291: "warning--falling-rocks-or-debris-right--g1",
     292: "warning--falling-rocks-or-debris-right--g2",
     293: "warning--falling-rocks-or-debris-right--g4",
     294: "warning--flaggers-in-road--g1",
     295: "warning--hairpin-curve-left--g1",
     296: "warning--hairpin-curve-left--g3",
     297: "warning--hairpin-curve-right--g1",
     298: "warning--hairpin-curve-right--g4",
     299: "warning--height-restriction--g2",
     300: "warning--horizontal-alignment-left--g1",
     301: "warning--horizontal-alignment-right--g1",
     302: "warning--horizontal-alignment-right--g3",
     303: "warning--junction-with-a-side-road-acute-left--g1",
     304: "warning--junction-with-a-side-road-acute-right--g1",
     305: "warning--junction-with-a-side-road-perpendicular-left--g1",
     306: "warning--junction-with-a-side-road-perpendicular-left--g3",
     307: "warning--junction-with-a-side-road-perpendicular-left--g4",
     308: "warning--junction-with-a-side-road-perpendicular-right--g1",
     309: "warning--junction-with-a-side-road-perpendicular-right--g3",
     310: "warning--kangaloo-crossing--g1",
     311: "warning--loop-270-degree--g1",
     312: "warning--narrow-bridge--g1",
     313: "warning--narrow-bridge--g3",
     314: "warning--offset-roads--g3",
     315: "warning--other-danger--g1",
     316: "warning--other-danger--g3",
     317: "warning--pass-left-or-right--g1",
     318: "warning--pass-left-or-right--g2",
     319: "warning--pedestrian-stumble-train--g1",
     320: "warning--pedestrians-crossing--g1",
     321: "warning--pedestrians-crossing--g10",
     322: "warning--pedestrians-crossing--g11",
     323: "warning--pedestrians-crossing--g12",
     324: "warning--pedestrians-crossing--g4",
     325: "warning--pedestrians-crossing--g5",
     326: "warning--pedestrians-crossing--g9",
     327: "warning--playground--g1",
     328: "warning--playground--g3",
     329: "warning--railroad-crossing--g1",
     330: "warning--railroad-crossing--g3",
     331: "warning--railroad-crossing--g4",
     332: "warning--railroad-crossing-with-barriers--g1",
     333: "warning--railroad-crossing-with-barriers--g2",
     334: "warning--railroad-crossing-with-barriers--g4",
     335: "warning--railroad-crossing-without-barriers--g1",
     336: "warning--railroad-crossing-without-barriers--g3",
     337: "warning--railroad-crossing-without-barriers--g4",
     338: "warning--railroad-intersection--g3",
     339: "warning--railroad-intersection--g4",
     340: "warning--restricted-zone--g1",
     341: "warning--road-bump--g1",
     342: "warning--road-bump--g2",
     343: "warning--road-narrows--g1",
     344: "warning--road-narrows--g2",
     345: "warning--road-narrows-left--g1",
     346: "warning--road-narrows-left--g2",
     347: "warning--road-narrows-right--g1",
     348: "warning--road-narrows-right--g2",
     349: "warning--road-widens--g1",
     350: "warning--road-widens-right--g1",
     351: "warning--roadworks--g1",
     352: "warning--roadworks--g2",
     353: "warning--roadworks--g3",
     354: "warning--roadworks--g4",
     355: "warning--roundabout--g1",
     356: "warning--roundabout--g25",
     357: "warning--school-zone--g2",
     358: "warning--shared-lane-motorcycles-bicycles--g1",
     359: "warning--slippery-motorcycles--g1",
     360: "warning--slippery-road-surface--g1",
     361: "warning--slippery-road-surface--g2",
     362: "warning--steep-ascent--g7",
     363: "warning--stop-ahead--g9",
     364: "warning--t-roads--g1",
     365: "warning--t-roads--g2",
     366: "warning--texts--g1",
     367: "warning--texts--g2",
     368: "warning--texts--g3",
     369: "warning--traffic-merges-left--g1",
     370: "warning--traffic-merges-left--g2",
     371: "warning--traffic-merges-right--g1",
     372: "warning--traffic-merges-right--g2",
     373: "warning--traffic-signals--g1",
     374: "warning--traffic-signals--g2",
     375: "warning--traffic-signals--g3",
     376: "warning--traffic-signals--g4",
     377: "warning--trail-crossing--g2",
     378: "warning--trams-crossing--g1",
     379: "warning--trucks-crossing--g1",
     380: "warning--turn-left--g1",
     381: "warning--turn-right--g1",
     382: "warning--turn-right--g2",
     383: "warning--two-way-traffic--g1",
     384: "warning--two-way-traffic--g2",
     385: "warning--uneven-road--g2",
     386: "warning--uneven-road--g6",
     387: "warning--uneven-roads-ahead--g1",
     388: "warning--wild-animals--g1",
     389: "warning--wild-animals--g4",
     390: "warning--winding-road-first-left--g1",
     391: "warning--winding-road-first-left--g2",
     392: "warning--winding-road-first-right--g1",
     393: "warning--winding-road-first-right--g3",
     394: "warning--wombat-crossing--g1",
     395: "warning--y-roads--g1"
     }
mtsd_category2label = {k: i for i, k in enumerate(mtsd_category2name.keys())}
mtsd_label2category = {v: k for k, v in mtsd_category2label.items()}
