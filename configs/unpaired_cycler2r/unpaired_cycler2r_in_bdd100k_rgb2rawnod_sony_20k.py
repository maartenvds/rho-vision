_base_ = [
    './runtime.py',
    '../_base_/datasets/bdd100k_rgb2rawnod_sony_512x512.py'
]

exp_name = 'unpaired_cycler2r_bdd100k_rgb2rawnod_sony'
work_dir = f'./work_dirs/experiments/{exp_name}'
