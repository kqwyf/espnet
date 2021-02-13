#!/usr/bin/env python3

# Copyright 2020 Shanghai Jiao Tong University (Chenda Li)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
from distutils.util import strtobool
import logging
import os

import kaldiio
import torch
import numpy as np

from espnet.utils.cli_utils import get_commandline_args
from espnet.utils.cli_writers import file_writer_helper
from espnet2.utils.types import int_or_none
from espnet2.fileio.video_scp import VideoScpReader, VideoScpWriter


def get_parser():
    parser = argparse.ArgumentParser(
        description="Pick parallel visual features for speech mixture",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")
    parser.add_argument("--min_or_max", default="min", type=str,
                        help="Mixture mode, 'min' or 'max' or 'fix', default min, if 'fix', the duration will be 2.4s",)

    parser.add_argument("rspecifier", type=str, help="input visual scp file")
    parser.add_argument("mixture_meta", type=str, help="mixture meta file")
    parser.add_argument("output_dir", type=str, help="vfeature output dir")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    assert args.min_or_max in ('min', 'max', 'fix')

    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    videos_raw = VideoScpReader(args.rspecifier, tofloat=False)
    writter_v1 = VideoScpWriter(f'{args.output_dir}/video1', f'{args.output_dir}/video1.scp')
    writter_v2 = VideoScpWriter(f'{args.output_dir}/video2', f'{args.output_dir}/video2.scp')
    with open(args.mixture_meta) as mixture_meta:
        for mix_key in mixture_meta.readlines():
            mix_key = mix_key.strip()
            meta = mix_key.split('_')
            key_1 = "_".join(meta[0:2])
            key_2 = "_".join(meta[2:4])

            v_1 , v_2 = videos_raw[key_1], videos_raw[key_2]

            len_1, len_2 = len(v_1), len(v_2)
            dims = v_1.shape[1:]

            if args.min_or_max == 'min':
                l = min(len_1, len_2)
                v_1, v_2 = v_1[0:l], v_2[0:l]
            elif args.min_or_max == 'max':
                l = max(len_1, len_2)
                v_1_, v_2_ = np.zeros((l, *dims), dtype=np.uint8), np.zeros((l, *dims), dtype=np.uint8)
                v_1_[0:len_1] += v_1
                v_2_[0:len_2] += v_2
                v_1, v_2 = v_1_, v_2_
            elif args.min_or_max == 'fix':
                v_1_, v_2_ = np.zeros((60, *dims), dtype=np.uint8), np.zeros((60, *dims), dtype=np.uint8)
                v_1_[0:len_1] += v_1[0:60]
                v_2_[0:len_2] += v_2[0:60]
                v_1, v_2 = v_1_, v_2_

            writter_v1[mix_key] = v_1
            writter_v2[mix_key] = v_2

    writter_v1.close()
    writter_v2.close()


if __name__ == "__main__":
    main()
