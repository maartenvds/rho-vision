_base_ = [
    './runtime.py',
    '../_base_/datasets/coco_rgb2pascalraw_512x512.py'
]

exp_name = 'unpaired_cycler2r_coco_rgb2pascalraw'
work_dir = f'./work_dirs/experiments/{exp_name}'
