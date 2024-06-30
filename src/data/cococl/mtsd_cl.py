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
        print("Task ID", self.task_idx, class_ids_current)
        buffer_ids = list(set(list(range(1, len(mtsd_category2name)+1))) - set(class_ids_current))

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
        # print("B: ", class_ids_current, self.coco.cats)
        for class_id in class_ids_current:
            try:
                cats[class_id] = self.coco.cats[class_id]
            except KeyError:
                pass
        self.coco.cats = cats
        print("After: ", cats)
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


mtsd_category2name = {1 : "regulatory--bicycles-only",
2 : "regulatory--pedestrians-only",
3 : "regulatory--buses-only",
4 : "regulatory--mopeds-and-bicycles-only",
5 : "regulatory--end-of-buses-only",
6 : "regulatory--end-of-bicycles-only",
7 : "regulatory--do-not-block-intersection",
8 : "regulatory--dual-lanes-go-straight-on-left",
9 : "regulatory--dual-lanes-go-straight-on-right",
10 : "regulatory--dual-path-bicycles-and-pedestrians",
11 : "regulatory--end-of-maximum-speed-limit-30",
12 : "regulatory--end-of-prohibition",
13 : "regulatory--end-of-priority-road",
14 : "regulatory--end-of-speed-limit-zone",
15 : "regulatory--end-of-no-parking",
16 : "regulatory--end-of-maximum-speed-limit-70",
17 : "regulatory--give-way-to-oncoming-traffic",
18 : "regulatory--go-straight",
19 : "regulatory--go-straight-or-turn-left",
20 : "regulatory--go-straight-or-turn-right",
21 : "regulatory--height-limit",
22 : "regulatory--keep-right",
23 : "regulatory--keep-left",
24 : "regulatory--lane-control",
25 : "regulatory--left-turn-yield-on-green",
26 : "regulatory--maximum-speed-limit-45",
27 : "regulatory--maximum-speed-limit-30",
28 : "regulatory--maximum-speed-limit-40",
29 : "regulatory--maximum-speed-limit-90",
30 : "regulatory--maximum-speed-limit-50",
31 : "regulatory--maximum-speed-limit-70",
32 : "regulatory--maximum-speed-limit-20",
33 : "regulatory--maximum-speed-limit-60",
34 : "regulatory--maximum-speed-limit-110",
35 : "regulatory--maximum-speed-limit-5",
36 : "regulatory--maximum-speed-limit-80",
37 : "regulatory--maximum-speed-limit-25",
38 : "regulatory--maximum-speed-limit-55",
39 : "regulatory--maximum-speed-limit-10",
40 : "regulatory--maximum-speed-limit-15",
41 : "regulatory--maximum-speed-limit-120",
42 : "regulatory--maximum-speed-limit-led-60",
43 : "regulatory--maximum-speed-limit-35",
44 : "regulatory--maximum-speed-limit-100",
45 : "regulatory--maximum-speed-limit-led-100",
46 : "regulatory--maximum-speed-limit-65",
47 : "regulatory--maximum-speed-limit-led-80",
48 : "regulatory--no-entry",
49 : "regulatory--no-stopping",
50 : "regulatory--no-parking",
51 : "regulatory--no-straight-through",
52 : "regulatory--no-heavy-goods-vehicles",
53 : "regulatory--no-left-turn",
54 : "regulatory--no-overtaking",
55 : "regulatory--no-turn-on-red",
56 : "regulatory--no-parking-or-no-stopping",
57 : "regulatory--no-vehicles-carrying-dangerous-goods",
58 : "regulatory--no-pedestrians",
59 : "regulatory--no-bicycles",
60 : "regulatory--no-motor-vehicle-trailers",
61 : "regulatory--no-stopping--g15",
62 : "regulatory--no-u-turn",
63 : "regulatory--no-motor-vehicles",
64 : "regulatory--no-motorcycles",
65 : "regulatory--no-buses",
66 : "regulatory--no-mopeds-or-bicycles",
67 : "regulatory--no-motor-vehicles-except-motorcycles",
68 : "regulatory--no-right-turn",
69 : "regulatory--no-overtaking-by-heavy-goods-vehicles",
70 : "regulatory--no-turns",
71 : "regulatory--no-hawkers",
72 : "regulatory--no-heavy-goods-vehicles-or-buses",
73 : "regulatory--no-pedestrians-or-bicycles",
74 : "regulatory--one-way-left",
75 : "regulatory--one-way-right",
76 : "regulatory--one-way-straight",
77 : "regulatory--parking-restrictions",
78 : "regulatory--passing-lane-ahead",
79 : "regulatory--pass-on-either-side",
80 : "regulatory--priority-over-oncoming-vehicles",
81 : "regulatory--priority-road",
82 : "regulatory--radar-enforced",
83 : "regulatory--reversible-lanes",
84 : "regulatory--road-closed",
85 : "regulatory--road-closed-to-vehicles",
86 : "regulatory--roundabout",
87 : "regulatory--shared-path-pedestrians-and-bicycles",
88 : "regulatory--shared-path-bicycles-and-pedestrians",
89 : "regulatory--stop",
90 : "regulatory--stop-here-on-red-or-flashing-light",
91 : "regulatory--stop-signals",
92 : "regulatory--triple-lanes-turn-left-center-lane",
93 : "regulatory--turn-right",
94 : "regulatory--turn-left",
95 : "regulatory--turn-right-ahead",
96 : "regulatory--turn-left-ahead",
97 : "regulatory--turning-vehicles-yield-to-pedestrians",
98 : "regulatory--u-turn",
99 : "regulatory--weight-limit",
100 : "regulatory--wrong-way",
101 : "regulatory--yield",
102 : "information--airport",
103 : "information--bike-route",
104 : "information--bus-stop",
105 : "information--children",
106 : "information--dead-end",
107 : "information--disabled-persons",
108 : "information--emergency-facility",
109 : "information--end-of-built-up-area",
110 : "information--end-of-motorway",
111 : "information--end-of-pedestrians-only",
112 : "information--end-of-living-street",
113 : "information--end-of-limited-access-road",
114 : "information--food",
115 : "information--gas-station",
116 : "information--highway-exit",
117 : "information--highway-interstate-route",
118 : "information--hospital",
119 : "information--interstate-route",
120 : "information--limited-access-road",
121 : "information--living-street",
122 : "information--motorway",
123 : "information--parking",
124 : "information--pedestrians-crossing",
125 : "information--road-bump",
126 : "information--safety-area",
127 : "information--telephone",
128 : "information--trailer-camping",
129 : "information--tram-bus-stop",
130 : "warning--added-lane-right",
131 : "warning--bicycles-crossing",
132 : "warning--children",
133 : "warning--crossroads",
134 : "warning--crossroads-with-priority-to-the-right",
135 : "warning--curve-left",
136 : "warning--curve-right",
137 : "warning--divided-highway-ends",
138 : "warning--domestic-animals",
139 : "warning--double-curve-first-left",
140 : "warning--double-turn-first-right",
141 : "warning--double-reverse-curve-right",
142 : "warning--double-curve-first-right",
143 : "warning--dual-lanes-right-turn-or-go-straight",
144 : "warning--emergency-vehicles",
145 : "warning--falling-rocks-or-debris-right",
146 : "warning--flaggers-in-road",
147 : "warning--hairpin-curve-right",
148 : "warning--height-restriction",
149 : "warning--horizontal-alignment-right",
150 : "warning--horizontal-alignment-left",
151 : "warning--junction-with-a-side-road-perpendicular-left",
152 : "warning--junction-with-a-side-road-acute-left",
153 : "warning--junction-with-a-side-road-perpendicular-right",
154 : "warning--junction-with-a-side-road-acute-right",
155 : "warning--kangaloo-crossing",
156 : "warning--narrow-bridge",
157 : "warning--other-danger",
158 : "warning--pass-left-or-right",
159 : "warning--pedestrians-crossing",
160 : "warning--railroad-crossing",
161 : "warning--railroad-crossing-with-barriers",
162 : "warning--railroad-crossing-without-barriers",
163 : "warning--railroad-intersection",
164 : "warning--road-bump",
165 : "warning--road-narrows",
166 : "warning--road-narrows-left",
167 : "warning--road-narrows-right",
168 : "warning--road-widens-right",
169 : "warning--road-widens",
170 : "warning--roadworks",
171 : "warning--roundabout",
172 : "warning--school-zone",
173 : "warning--slippery-road-surface",
174 : "warning--steep-ascent",
175 : "warning--stop-ahead",
176 : "warning--texts",
177 : "warning--traffic-merges-right",
178 : "warning--traffic-merges-left",
179 : "warning--traffic-signals",
180 : "warning--trail-crossing",
181 : "warning--t-roads",
182 : "warning--trucks-crossing",
183 : "warning--turn-right",
184 : "warning--turn-left",
185 : "warning--two-way-traffic",
186 : "warning--uneven-road",
187 : "warning--wild-animals",
188 : "warning--winding-road-first-left",
189 : "warning--winding-road-first-right",
190 : "warning--y-roads",
191 : "complementary--both-directions",
192 : "complementary--buses",
193 : "complementary--chevron-left",
194 : "complementary--chevron-right-unsure",
195 : "complementary--chevron-right",
196 : "complementary--distance",
197 : "complementary--except-bicycles",
198 : "complementary--go-left",
199 : "complementary--go-right",
200 : "complementary--keep-right",
201 : "complementary--keep-left",
202 : "complementary--maximum-speed-limit-70",
203 : "complementary--maximum-speed-limit-30",
204 : "complementary--maximum-speed-limit-20",
205 : "complementary--maximum-speed-limit-55",
206 : "complementary--maximum-speed-limit-45",
207 : "complementary--maximum-speed-limit-75",
208 : "complementary--maximum-speed-limit-40",
209 : "complementary--maximum-speed-limit-25",
210 : "complementary--maximum-speed-limit-15",
211 : "complementary--maximum-speed-limit-35",
212 : "complementary--maximum-speed-limit-50",
213 : "complementary--obstacle-delineator",
214 : "complementary--one-direction-right",
215 : "complementary--one-direction-left",
216 : "complementary--pass-right",
217 : "complementary--tow-away-zone",
218 : "complementary--trucks",
219 : "complementary--trucks-turn-right",
220 : "complementary--turn-left",
221 : "complementary--turn-right",
222 : "warning--steep-descent",
223 : "complementary--accident-area",
224 : "complementary--accident-area--g3",
225 : "complementary--both-directions--g1",
226 : "complementary--buses--g1",
227 : "complementary--chevron-left--g1",
228 : "complementary--chevron-left--g2",
229 : "complementary--chevron-left--g3",
230 : "complementary--chevron-left--g4",
231 : "complementary--chevron-left--g5",
232 : "complementary--chevron-right--g1",
233 : "complementary--chevron-right--g3",
234 : "complementary--chevron-right--g4",
235 : "complementary--chevron-right--g5",
236 : "complementary--chevron-right-unsure--g6",
237 : "complementary--distance--g1",
238 : "complementary--distance--g2",
239 : "complementary--distance--g3",
240 : "complementary--except-bicycles--g1",
241 : "complementary--extent-of-prohibition-area-both-direction",
242 : "complementary--extent-of-prohibition-area-both-direction--g1",
243 : "complementary--go-left--g1",
244 : "complementary--go-right--g1",
245 : "complementary--go-right--g2",
246 : "complementary--keep-left--g1",
247 : "complementary--keep-right--g1",
248 : "complementary--maximum-speed-limit-15--g1",
249 : "complementary--maximum-speed-limit-20--g1",
250 : "complementary--maximum-speed-limit-25--g1",
251 : "complementary--maximum-speed-limit-30--g1",
252 : "complementary--maximum-speed-limit-35--g1",
253 : "complementary--maximum-speed-limit-40--g1",
254 : "complementary--maximum-speed-limit-45--g1",
255 : "complementary--maximum-speed-limit-50--g1",
256 : "complementary--maximum-speed-limit-55--g1",
257 : "complementary--maximum-speed-limit-70--g1",
258 : "complementary--maximum-speed-limit-75--g1",
259 : "complementary--obstacle-delineator--g1",
260 : "complementary--obstacle-delineator--g2",
261 : "complementary--one-direction-left--g1",
262 : "complementary--one-direction-right--g1",
263 : "complementary--pass-right--g1",
264 : "complementary--priority-route-at-intersection",
265 : "complementary--priority-route-at-intersection--g1",
266 : "complementary--tow-away-zone--g1",
267 : "complementary--trucks--g1",
268 : "complementary--trucks-turn-right--g1",
269 : "complementary--turn-left--g2",
270 : "complementary--turn-right--g2",
271 : "information--airport--g1",
272 : "information--airport--g2",
273 : "information--bike-route--g1",
274 : "information--bus-stop--g1",
275 : "information--camp",
276 : "information--camp--g1",
277 : "information--central-lane",
278 : "information--central-lane--g1",
279 : "information--dead-end--g1",
280 : "information--dead-end-except-bicycles",
281 : "information--dead-end-except-bicycles--g1",
282 : "information--disabled-persons--g1",
283 : "information--emergency-facility--g2",
284 : "information--end-of-built-up-area--g1",
285 : "information--end-of-limited-access-road--g1",
286 : "information--end-of-living-street--g1",
287 : "information--end-of-motorway--g1",
288 : "information--end-of-pedestrians-only--g2",
289 : "information--food--g2",
290 : "information--gas-station--g1",
291 : "information--highway-exit--g1",
292 : "information--hospital--g1",
293 : "information--interstate-route--g1",
294 : "information--limited-access-road--g1",
295 : "information--living-street--g1",
296 : "information--lodging",
297 : "information--lodging--g1",
298 : "information--minimum-speed-40",
299 : "information--minimum-speed-40--g1",
300 : "information--motorway--g1",
301 : "information--no-parking",
302 : "information--no-parking--g3",
303 : "information--parking--g1",
304 : "information--parking--g2",
305 : "information--parking--g3",
306 : "information--parking--g45",
307 : "information--parking--g5",
308 : "information--parking--g6",
309 : "information--pedestrians-crossing--g1",
310 : "information--pedestrians-crossing--g2",
311 : "information--road-bump--g1",
312 : "information--safety-area--g2",
313 : "information--stairs",
314 : "information--stairs--g1",
315 : "information--telephone--g1",
316 : "information--telephone--g2",
317 : "information--trailer-camping--g1",
318 : "other-sign",
319 : "regulatory--bicycles-only--g1",
320 : "regulatory--bicycles-only--g2",
321 : "regulatory--bicycles-only--g3",
322 : "regulatory--buses-only--g1",
323 : "regulatory--detour-left",
324 : "regulatory--detour-left--g1",
325 : "regulatory--do-not-block-intersection--g1",
326 : "regulatory--do-not-stop-on-tracks",
327 : "regulatory--do-not-stop-on-tracks--g1",
328 : "regulatory--dual-lanes-go-straight-on-left--g1",
329 : "regulatory--dual-lanes-go-straight-on-right--g1",
330 : "regulatory--dual-lanes-turn-left-no-u-turn",
331 : "regulatory--dual-lanes-turn-left-no-u-turn--g1",
332 : "regulatory--dual-lanes-turn-left-or-straight",
333 : "regulatory--dual-lanes-turn-left-or-straight--g1",
334 : "regulatory--dual-lanes-turn-right-or-straight",
335 : "regulatory--dual-lanes-turn-right-or-straight--g1",
336 : "regulatory--dual-path-bicycles-and-pedestrians--g1",
337 : "regulatory--dual-path-pedestrians-and-bicycles",
338 : "regulatory--dual-path-pedestrians-and-bicycles--g1",
339 : "regulatory--end-of-bicycles-only--g1",
340 : "regulatory--end-of-buses-only--g1",
341 : "regulatory--end-of-maximum-speed-limit-30--g2",
342 : "regulatory--end-of-maximum-speed-limit-70--g1",
343 : "regulatory--end-of-maximum-speed-limit-70--g2",
344 : "regulatory--end-of-no-parking--g1",
345 : "regulatory--end-of-priority-road--g1",
346 : "regulatory--end-of-prohibition--g1",
347 : "regulatory--end-of-speed-limit-zone--g1",
348 : "regulatory--give-way-to-oncoming-traffic--g1",
349 : "regulatory--go-straight--g1",
350 : "regulatory--go-straight--g3",
351 : "regulatory--go-straight-or-turn-left--g1",
352 : "regulatory--go-straight-or-turn-left--g2",
353 : "regulatory--go-straight-or-turn-left--g3",
354 : "regulatory--go-straight-or-turn-right--g1",
355 : "regulatory--go-straight-or-turn-right--g3",
356 : "regulatory--height-limit--g1",
357 : "regulatory--keep-left--g1",
358 : "regulatory--keep-left--g2",
359 : "regulatory--keep-right--g1",
360 : "regulatory--keep-right--g2",
361 : "regulatory--keep-right--g4",
362 : "regulatory--keep-right--g6",
363 : "regulatory--lane-control--g1",
364 : "regulatory--left-turn-yield-on-green--g1",
365 : "regulatory--maximum-speed-limit-10--g1",
366 : "regulatory--maximum-speed-limit-100--g1",
367 : "regulatory--maximum-speed-limit-100--g3",
368 : "regulatory--maximum-speed-limit-110--g1",
369 : "regulatory--maximum-speed-limit-120--g1",
370 : "regulatory--maximum-speed-limit-15--g1",
371 : "regulatory--maximum-speed-limit-20--g1",
372 : "regulatory--maximum-speed-limit-25--g1",
373 : "regulatory--maximum-speed-limit-25--g2",
374 : "regulatory--maximum-speed-limit-30--g1",
375 : "regulatory--maximum-speed-limit-30--g3",
376 : "regulatory--maximum-speed-limit-35--g2",
377 : "regulatory--maximum-speed-limit-40--g1",
378 : "regulatory--maximum-speed-limit-40--g3",
379 : "regulatory--maximum-speed-limit-40--g6",
380 : "regulatory--maximum-speed-limit-45--g1",
381 : "regulatory--maximum-speed-limit-45--g3",
382 : "regulatory--maximum-speed-limit-5--g1",
383 : "regulatory--maximum-speed-limit-50--g1",
384 : "regulatory--maximum-speed-limit-50--g6",
385 : "regulatory--maximum-speed-limit-55--g2",
386 : "regulatory--maximum-speed-limit-60--g1",
387 : "regulatory--maximum-speed-limit-65--g2",
388 : "regulatory--maximum-speed-limit-70--g1",
389 : "regulatory--maximum-speed-limit-80--g1",
390 : "regulatory--maximum-speed-limit-90--g1",
391 : "regulatory--maximum-speed-limit-led-100--g1",
392 : "regulatory--maximum-speed-limit-led-60--g1",
393 : "regulatory--maximum-speed-limit-led-80--g1",
394 : "regulatory--minimum-safe-distance",
395 : "regulatory--minimum-safe-distance--g1",
396 : "regulatory--mopeds-and-bicycles-only--g1",
397 : "regulatory--no-bicycles--g1",
398 : "regulatory--no-bicycles--g2",
399 : "regulatory--no-bicycles--g3",
400 : "regulatory--no-buses--g3",
401 : "regulatory--no-entry--g1",
402 : "regulatory--no-hawkers--g1",
403 : "regulatory--no-heavy-goods-vehicles--g1",
404 : "regulatory--no-heavy-goods-vehicles--g2",
405 : "regulatory--no-heavy-goods-vehicles--g4",
406 : "regulatory--no-heavy-goods-vehicles--g5",
407 : "regulatory--no-heavy-goods-vehicles-or-buses--g1",
408 : "regulatory--no-left-turn--g1",
409 : "regulatory--no-left-turn--g2",
410 : "regulatory--no-left-turn--g3",
411 : "regulatory--no-mopeds-or-bicycles--g1",
412 : "regulatory--no-motor-vehicle-trailers--g1",
413 : "regulatory--no-motor-vehicles--g1",
414 : "regulatory--no-motor-vehicles--g4",
415 : "regulatory--no-motor-vehicles-except-motorcycles--g1",
416 : "regulatory--no-motor-vehicles-except-motorcycles--g2",
417 : "regulatory--no-motorcycles--g1",
418 : "regulatory--no-motorcycles--g2",
419 : "regulatory--no-overtaking--g1",
420 : "regulatory--no-overtaking--g2",
421 : "regulatory--no-overtaking--g4",
422 : "regulatory--no-overtaking--g5",
423 : "regulatory--no-overtaking-by-heavy-goods-vehicles--g1",
424 : "regulatory--no-parking--g1",
425 : "regulatory--no-parking--g2",
426 : "regulatory--no-parking--g5",
427 : "regulatory--no-parking-or-no-stopping--g1",
428 : "regulatory--no-parking-or-no-stopping--g2",
429 : "regulatory--no-parking-or-no-stopping--g3",
430 : "regulatory--no-pedestrians--g1",
431 : "regulatory--no-pedestrians--g2",
432 : "regulatory--no-pedestrians--g3",
433 : "regulatory--no-pedestrians-or-bicycles--g1",
434 : "regulatory--no-right-turn--g1",
435 : "regulatory--no-right-turn--g2",
436 : "regulatory--no-right-turn--g3",
437 : "regulatory--no-stopping--g2",
438 : "regulatory--no-stopping--g4",
439 : "regulatory--no-stopping--g5",
440 : "regulatory--no-stopping--g8",
441 : "regulatory--no-straight-through--g1",
442 : "regulatory--no-straight-through--g2",
443 : "regulatory--no-turn-on-red--g1",
444 : "regulatory--no-turn-on-red--g2",
445 : "regulatory--no-turn-on-red--g3",
446 : "regulatory--no-turns--g1",
447 : "regulatory--no-u-turn--g1",
448 : "regulatory--no-u-turn--g2",
449 : "regulatory--no-u-turn--g3",
450 : "regulatory--no-vehicles-carrying-dangerous-goods--g1",
451 : "regulatory--one-way-left--g1",
452 : "regulatory--one-way-left--g2",
453 : "regulatory--one-way-left--g3",
454 : "regulatory--one-way-right--g1",
455 : "regulatory--one-way-right--g2",
456 : "regulatory--one-way-right--g3",
457 : "regulatory--one-way-straight--g1",
458 : "regulatory--parking-restrictions--g2",
459 : "regulatory--pass-on-either-side--g1",
460 : "regulatory--pass-on-either-side--g2",
461 : "regulatory--passing-lane-ahead--g1",
462 : "regulatory--pedestrians-only--g1",
463 : "regulatory--pedestrians-only--g2",
464 : "regulatory--priority-over-oncoming-vehicles--g1",
465 : "regulatory--priority-road--g4",
466 : "regulatory--radar-enforced--g1",
467 : "regulatory--reversible-lanes--g2",
468 : "regulatory--road-closed--g1",
469 : "regulatory--road-closed--g2",
470 : "regulatory--road-closed-to-vehicles--g1",
471 : "regulatory--road-closed-to-vehicles--g3",
472 : "regulatory--roundabout--g1",
473 : "regulatory--roundabout--g2",
474 : "regulatory--shared-path-bicycles-and-pedestrians--g1",
475 : "regulatory--shared-path-pedestrians-and-bicycles--g1",
476 : "regulatory--stop--g1",
477 : "regulatory--stop--g10",
478 : "regulatory--stop--g2",
479 : "regulatory--stop-here-on-red-or-flashing-light--g1",
480 : "regulatory--stop-here-on-red-or-flashing-light--g2",
481 : "regulatory--stop-signals--g1",
482 : "regulatory--text-four-lines",
483 : "regulatory--text-four-lines--g1",
484 : "regulatory--triple-lanes-turn-left-center-lane--g1",
485 : "regulatory--truck-speed-limit-60",
486 : "regulatory--truck-speed-limit-60--g1",
487 : "regulatory--turn-left--g1",
488 : "regulatory--turn-left--g2",
489 : "regulatory--turn-left--g3",
490 : "regulatory--turn-left-ahead--g1",
491 : "regulatory--turn-right--g1",
492 : "regulatory--turn-right--g2",
493 : "regulatory--turn-right--g3",
494 : "regulatory--turn-right-ahead--g1",
495 : "regulatory--turn-right-ahead--g2",
496 : "regulatory--turning-vehicles-yield-to-pedestrians--g1",
497 : "regulatory--u-turn--g1",
498 : "regulatory--weight-limit--g1",
499 : "regulatory--weight-limit-with-trucks",
500 : "regulatory--weight-limit-with-trucks--g1",
501 : "regulatory--width-limit",
502 : "regulatory--width-limit--g1",
503 : "regulatory--wrong-way--g1",
504 : "regulatory--yield--g1",
505 : "warning--accidental-area-unsure",
506 : "warning--accidental-area-unsure--g2",
507 : "warning--added-lane-right--g1",
508 : "warning--bicycles-crossing--g1",
509 : "warning--bicycles-crossing--g2",
510 : "warning--bicycles-crossing--g3",
511 : "warning--bus-stop-ahead",
512 : "warning--bus-stop-ahead--g3",
513 : "warning--children--g1",
514 : "warning--children--g2",
515 : "warning--crossroads--g1",
516 : "warning--crossroads--g3",
517 : "warning--crossroads-with-priority-to-the-right--g1",
518 : "warning--curve-left--g1",
519 : "warning--curve-left--g2",
520 : "warning--curve-right--g1",
521 : "warning--curve-right--g2",
522 : "warning--dip",
523 : "warning--dip--g2",
524 : "warning--divided-highway-ends--g1",
525 : "warning--divided-highway-ends--g2",
526 : "warning--domestic-animals--g1",
527 : "warning--domestic-animals--g3",
528 : "warning--double-curve-first-left--g1",
529 : "warning--double-curve-first-left--g2",
530 : "warning--double-curve-first-right--g1",
531 : "warning--double-curve-first-right--g2",
532 : "warning--double-reverse-curve-right--g1",
533 : "warning--double-turn-first-right--g1",
534 : "warning--dual-lanes-right-turn-or-go-straight--g1",
535 : "warning--emergency-vehicles--g1",
536 : "warning--equestrians-crossing",
537 : "warning--equestrians-crossing--g2",
538 : "warning--falling-rocks-or-debris-right--g1",
539 : "warning--falling-rocks-or-debris-right--g2",
540 : "warning--falling-rocks-or-debris-right--g4",
541 : "warning--flaggers-in-road--g1",
542 : "warning--hairpin-curve-left",
543 : "warning--hairpin-curve-left--g1",
544 : "warning--hairpin-curve-left--g3",
545 : "warning--hairpin-curve-right--g1",
546 : "warning--hairpin-curve-right--g4",
547 : "warning--height-restriction--g2",
548 : "warning--horizontal-alignment-left--g1",
549 : "warning--horizontal-alignment-right--g1",
550 : "warning--horizontal-alignment-right--g3",
551 : "warning--junction-with-a-side-road-acute-left--g1",
552 : "warning--junction-with-a-side-road-acute-right--g1",
553 : "warning--junction-with-a-side-road-perpendicular-left--g1",
554 : "warning--junction-with-a-side-road-perpendicular-left--g3",
555 : "warning--junction-with-a-side-road-perpendicular-left--g4",
556 : "warning--junction-with-a-side-road-perpendicular-right--g1",
557 : "warning--junction-with-a-side-road-perpendicular-right--g3",
558 : "warning--kangaloo-crossing--g1",
559 : "warning--loop-270-degree",
560 : "warning--loop-270-degree--g1",
561 : "warning--narrow-bridge--g1",
562 : "warning--narrow-bridge--g3",
563 : "warning--offset-roads",
564 : "warning--offset-roads--g3",
565 : "warning--other-danger--g1",
566 : "warning--other-danger--g3",
567 : "warning--pass-left-or-right--g1",
568 : "warning--pass-left-or-right--g2",
569 : "warning--pedestrian-stumble-train",
570 : "warning--pedestrian-stumble-train--g1",
571 : "warning--pedestrians-crossing--g1",
572 : "warning--pedestrians-crossing--g10",
573 : "warning--pedestrians-crossing--g11",
574 : "warning--pedestrians-crossing--g12",
575 : "warning--pedestrians-crossing--g4",
576 : "warning--pedestrians-crossing--g5",
577 : "warning--pedestrians-crossing--g9",
578 : "warning--playground",
579 : "warning--playground--g1",
580 : "warning--playground--g3",
581 : "warning--railroad-crossing--g1",
582 : "warning--railroad-crossing--g3",
583 : "warning--railroad-crossing--g4",
584 : "warning--railroad-crossing-with-barriers--g1",
585 : "warning--railroad-crossing-with-barriers--g2",
586 : "warning--railroad-crossing-with-barriers--g4",
587 : "warning--railroad-crossing-without-barriers--g1",
588 : "warning--railroad-crossing-without-barriers--g3",
589 : "warning--railroad-crossing-without-barriers--g4",
590 : "warning--railroad-intersection--g3",
591 : "warning--railroad-intersection--g4",
592 : "warning--restricted-zone",
593 : "warning--restricted-zone--g1",
594 : "warning--road-bump--g1",
595 : "warning--road-bump--g2",
596 : "warning--road-narrows--g1",
597 : "warning--road-narrows--g2",
598 : "warning--road-narrows-left--g1",
599 : "warning--road-narrows-left--g2",
600 : "warning--road-narrows-right--g1",
601 : "warning--road-narrows-right--g2",
602 : "warning--road-widens--g1",
603 : "warning--road-widens-right--g1",
604 : "warning--roadworks--g1",
605 : "warning--roadworks--g2",
606 : "warning--roadworks--g3",
607 : "warning--roadworks--g4",
608 : "warning--roundabout--g1",
609 : "warning--roundabout--g25",
610 : "warning--school-zone--g2",
611 : "warning--shared-lane-motorcycles-bicycles",
612 : "warning--shared-lane-motorcycles-bicycles--g1",
613 : "warning--slippery-motorcycles",
614 : "warning--slippery-motorcycles--g1",
615 : "warning--slippery-road-surface--g1",
616 : "warning--slippery-road-surface--g2",
617 : "warning--steep-ascent--g7",
618 : "warning--stop-ahead--g9",
619 : "warning--t-roads--g1",
620 : "warning--t-roads--g2",
621 : "warning--texts--g1",
622 : "warning--texts--g2",
623 : "warning--texts--g3",
624 : "warning--traffic-merges-left--g1",
625 : "warning--traffic-merges-left--g2",
626 : "warning--traffic-merges-right--g1",
627 : "warning--traffic-merges-right--g2",
628 : "warning--traffic-signals--g1",
629 : "warning--traffic-signals--g2",
630 : "warning--traffic-signals--g3",
631 : "warning--traffic-signals--g4",
632 : "warning--trail-crossing--g2",
633 : "warning--trams-crossing",
634 : "warning--trams-crossing--g1",
635 : "warning--trucks-crossing--g1",
636 : "warning--turn-left--g1",
637 : "warning--turn-right--g1",
638 : "warning--turn-right--g2",
639 : "warning--two-way-traffic--g1",
640 : "warning--two-way-traffic--g2",
641 : "warning--uneven-road--g2",
642 : "warning--uneven-road--g6",
643 : "warning--uneven-roads-ahead",
644 : "warning--uneven-roads-ahead--g1",
645 : "warning--wild-animals--g1",
646 : "warning--wild-animals--g4",
647 : "warning--winding-road-first-left--g1",
648 : "warning--winding-road-first-left--g2",
649 : "warning--winding-road-first-right--g1",
650 : "warning--winding-road-first-right--g3",
651 : "warning--wombat-crossing",
652 : "warning--wombat-crossing--g1",
653 : "warning--y-roads--g1",
}
mtsd_category2label = {k: i for i, k in enumerate(mtsd_category2name.keys())}
mtsd_label2category = {v: k for k, v in mtsd_category2label.items()}
