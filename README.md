- [Normal training](#normal-training)
- [CL Training](#cl-training)
  - [Contributors](#contributors)
  
# Normal training
1. Training 
   
   * `python scripts/train.py -t /path/to/ckpt`

2. Evaluate 
   
   * `python scripts/train.py -r /path/to/ckpt/training --test-only`


# CL Training 
1. Choose data ratio (`4040`, `7010`)
2. Normal training for first task `t` classes 
3. For next task
   1. Load checkpoint trained from previous task (manual)
   2. Config `class_ids`, `buffer_ids` on `coco_cache.py`
   3. Set `True` for `fake_query` and `distill_attn`
   4. Training with same scripts above

## Contributors 
<a href="https://github.com/tuanlda78202/MLR/graphs/contributors">
<img src="https://contrib.rocks/image?repo=tuanlda78202/MLR" /></a>
</a>