#!/usr/bin/env python3

# Copyright 2020 Shanghai Jiao Tong University (Chenda Li)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
from distutils.util import strtobool
import logging

import kaldiio
import torch
import numpy as np

import skvideo.io
from espnet.utils.cli_utils import get_commandline_args
from espnet.utils.cli_writers import file_writer_helper
from espnet2.utils.types import int_or_none
from espnet2.fileio.sound_scp import SoundScpWriter
import librosa



def get_parser():
    parser = argparse.ArgumentParser(
        description="convert mp4 to kaldi ark for LRS2 dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")

    parser.add_argument("rspecifier", type=str, help="input video scp file")
    parser.add_argument("wspecifier", type=str, help="output specifier")
    parser.add_argument("output_wav", type=str, help="output wav dir")

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()


    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())

    with open(args.rspecifier) as inputfile, \
            kaldiio.WriteHelper(args.wspecifier, compression_method=2) as writer_v, \
            SoundScpWriter(f"{args.output_wav}/wav", f"{args.output_wav}/wav.scp") as writer_a:
        for line in inputfile:
            line = line.strip()
            key, v_path = line.split()
            video = skvideo.io.vread(v_path)/255
            video = video.astype('float32')
            len, w, h, c = video.shape
            video = video.reshape(len, w * h * c)
            print(video.shape)
            writer_v[key] = video.astype('float32')

            audio, sr = librosa.load(v_path, 16000)
            writer_a[key] = sr, audio


  

if __name__ == "__main__":
    main()
