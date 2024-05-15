import os
import cv2
import glob
from tqdm import tqdm
import argparse
import numpy as np
from collections import OrderedDict
from imageio import imread, imwrite
from process_raw import DngFile

import torch
from torch import nn
import torch.distributions as D

from mmgen.models import build_module
from unpaired_cycler2r.models import *


class DemoModel(nn.Module):

    def __init__(self, ckpt_path) -> None:
        super().__init__()
        # load checkpoint
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            if k.startswith('generator.'):
                state_dict[k[len('generator.'):]] = v

        # get invISP model
        self.model = build_module(dict(type='inverseISP'))
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def _get_illumination_condition(self, img):
        mean_var = self.model.color_condition_gen(img)
        m = D.Normal(mean_var[:, 0],
                     torch.clamp_min(torch.abs(mean_var[:, 1]), 1e-6))
        color_condition = m.sample()
        mean_var = self.model.bright_condition_gen(img)
        m = D.Normal(mean_var[:, 0],
                     torch.clamp_min(torch.abs(mean_var[:, 1]), 1e-6))
        bright_condition = m.sample()
        condition = torch.cat(
            [color_condition[:, None], bright_condition[:, None]], 1)
        return condition

    def _mosaic(self, x):
        """Convert RGB to BAYER
        """
        h, w = x.shape[2:]
        _x = torch.zeros(x.shape[0], 2*h, 2*w, device=x.device)
        _x[:, 0::2, 0::2] = x[:, 0]     # R
        _x[:, 0::2, 1::2] = x[:, 1]     # G
        _x[:, 1::2, 0::2] = x[:, 1]     # G
        _x[:, 1::2, 1::2] = x[:, 2]     # B
        return _x

    def forward(self, rgb, mosaic=False):
        with torch.no_grad():
            # get illumination condition
            condition = self._get_illumination_condition(rgb)
            # get simulated RAW image
            raw = self.model(rgb, condition, rev=False)
            raw = torch.clamp(raw, 0, 1)
            if mosaic:
                raw = self._mosaic(raw)
        return raw


if __name__ == '__main__':
    args = argparse.ArgumentParser("Convert a folder with RGB images to a folder with simulated raw images")
    args.add_argument('ckpt', help="Checkpoint (weights) file of CycleR2R")
    args.add_argument('dataset', help="Path to the RGB dataset folder")
    args.add_argument('output', help="Output folder to store converted images in")
    args.add_argument('--bits', help="Number of bits per pixel", default=12, type=int)
    args.add_argument('--half', help="Reduce output resolution of the images by 2", action="store_true")
    args.add_argument('--mosaic', help="Do BAYER mosaic at the end. This will upscale the image by a factor of 2 in size", action="store_true")
    args = args.parse_args()

    assert args.bits <= 16
    max_value = 2**args.bits
    model = DemoModel(args.ckpt)
    model = model.cuda()

    files = glob.glob(os.path.join(args.dataset, '*'))

    for imgfile in tqdm(files):
        # read and preprocess image
        img = imread(imgfile)

        if args.half:
            assert not args.mosaic
            img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))

        img = img.astype(np.float32) / 255
        img = torch.from_numpy(img).permute(2, 0, 1)[None]
        img = img.cuda()

        # convert
        raw = model(img, mosaic=args.mosaic)

        # post-process
        raw = raw[0].cpu().numpy()
        raw = np.clip((raw * max_value), 0, max_value).astype(np.uint16)

        # save to DNG
        if args.mosaic:
            outfile = os.path.join(args.output, os.path.basename(imgfile).split('.')[0] + '.DNG')
            DngFile.save(outfile, raw, bit=args.bits, pattern='RGGB')
        else:
            # save as raw 16-bit RGB as png
            raw = raw.transpose(1, 2, 0)
            outfile = os.path.join(args.output, os.path.basename(imgfile).split('.')[0] + '.png')
            cv2.imwrite(outfile, raw)
