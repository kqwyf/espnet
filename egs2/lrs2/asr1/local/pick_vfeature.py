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


def get_parser():
    parser = argparse.ArgumentParser(
        description="Pick parallel visual features for speech mixture",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")
    parser.add_argument("--output_name", default="v", type=str)
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

    vfeatures = kaldiio.load_scp(args.rspecifier)
    writter_v1 = kaldiio.WriteHelper(
        f"ark,scp:{args.output_dir}/{args.output_name}1.ark,{args.output_dir}/{args.output_name}1.scp", compression_method=2)
    writter_v2 = kaldiio.WriteHelper(
        f"ark,scp:{args.output_dir}/{args.output_name}2.ark,{args.output_dir}/{args.output_name}2.scp", compression_method=2)

    with open(args.mixture_meta) as mixture_meta:
        for mix_key in mixture_meta.readlines():
            mix_key = mix_key.strip()
            meta = mix_key.split('_')
            key_1 = "_".join(meta[0:2])
            key_2 = "_".join(meta[2:4])

            try:
                v_1 , v_2 = vfeatures[key_1], vfeatures[key_2]
            except:
                print(f'key {key_1}, {key_2} not exist')
                continue

            len_1, len_2 = len(v_1), len(v_2)
            dim = v_1.shape[1]

            if args.min_or_max == 'min':
                l = min(len_1, len_2)
                v_1, v_2 = v_1[0:l], v_2[0:l]
            elif args.min_or_max == 'max':
                l = max(len_1, len_2)
                v_1_, v_2_ = np.zeros((l, dim)), np.zeros((l, dim))
                v_1_[0:len_1] += v_1
                v_2_[0:len_2] += v_2
                v_1, v_2 = v_1_, v_2_
            elif args.min_or_max == 'fix':
                v_1_, v_2_ = np.zeros((60, dim)), np.zeros((60, dim))
                v_1_[0:len_1] += v_1[0:60]
                v_2_[0:len_2] += v_2[0:60]
                v_1, v_2 = v_1_, v_2_

            writter_v1[mix_key] = v_1
            writter_v2[mix_key] = v_2

    writter_v1.close()
    writter_v2.close()


if __name__ == "__main__":
    main()
