#!/usr/bin/env python3

# Copyright 2020 Shanghai Jiao Tong University (Chenda Li)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
from distutils.util import strtobool
import logging

import kaldiio
import torch
import numpy as np

from espnet.utils.cli_utils import get_commandline_args
from espnet.utils.cli_writers import file_writer_helper
from espnet2.utils.types import int_or_none
from espnet2.fileio.sound_scp import SoundScpWriter


def get_parser():
    parser = argparse.ArgumentParser(
        description="create speech mixture for LRS2 dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")
    parser.add_argument("--min_or_max", default="min", type=str,
                        help="Mixture mode, 'min' or 'max' or 'fix', default min, if 'fix', the duration will be 2.4s",)

    parser.add_argument("rspecifier_wav", type=str, help="input audio scp file")
    parser.add_argument("rspecifier_text", type=str, help="input transcription file")
    parser.add_argument("mixture_meta", type=str, help="mixture meta file")

    parser.add_argument("output_wav", type=str, help="wav output dir")
    parser.add_argument("output_scp", type=str, help="scp output dir")
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

    sources = kaldiio.load_scp(args.rspecifier_wav)
    with open(args.rspecifier_text, 'r', encoding='utf-8') as text:
        text_sources = dict()
        for line in text:
            key, trans = line.split(maxsplit=1)
            text_sources[key] = trans
    writter_mix = SoundScpWriter(f"{args.output_wav}/mix", f"{args.output_scp}/wav.scp")
    writter_spk1 = SoundScpWriter(f"{args.output_wav}/spk1", f"{args.output_scp}/spk1.scp")
    writter_spk2 = SoundScpWriter(f"{args.output_wav}/spk2", f"{args.output_scp}/spk2.scp")
    writter_text_spk1 = open(f"{args.output_scp}/text_spk1", 'w', encoding='utf-8')
    writter_text_spk2 = open(f"{args.output_scp}/text_spk2", 'w', encoding='utf-8')

    with open(args.mixture_meta) as mixture_meta:
        for mix_key in mixture_meta.readlines():
            mix_key = mix_key.strip()
            meta = mix_key.split('_')
            key_1 = "_".join(meta[0:3])
            key_2 = "_".join(meta[3:6])
            snr_1, snr_2 = float(meta[6]), float(meta[7])
            (_, wav_1) , (_, wav_2) = sources[key_1], sources[key_2]

            # To prevent overflow
            wav_1 = wav_1 / wav_1.max()
            wav_2 = wav_2 / wav_2.max()

            # Power normalization
            wav_1 = wav_1 / np.sqrt(np.mean(wav_1 * wav_1))
            wav_2 = wav_2 / np.sqrt(np.mean(wav_2 * wav_2))

            # apply weight
            weight_1, weight_2 = 10 ** (snr_1 / 20), 10 ** (snr_2 / 20)
            wav_1, wav_2 = wav_1 * weight_1, wav_2 * weight_2
            peak = max(max(wav_1), max(wav_2))

            wav_1, wav_2 = 0.5 * wav_1 / peak, 0.5 * wav_2 / peak

            len_1, len_2 = len(wav_1), len(wav_2)

            if args.min_or_max == 'min':
                l = min(len_1, len_2)
                wav_1, wav_2 = wav_1[0:l], wav_2[0:l]
            elif args.min_or_max == 'max':
                l = max(len_1, len_2)
                wav_1_, wav_2_ = np.zeros(l), np.zeros(l)
                wav_1_[0:len_1] += wav_1
                wav_2_[0:len_2] += wav_2
                wav_1, wav_2 = wav_1_, wav_2_
            elif args.min_or_max == 'fix':
                wav_1_, wav_2_ = np.zeros(38400), np.zeros(38400)
                wav_1_[0:len_1] += wav_1[0:38400]
                wav_2_[0:len_2] += wav_2[0:38400]
                wav_1, wav_2 = wav_1_, wav_2_


            mixture = wav_1 + wav_2

            writter_spk1[mix_key] = 16000, wav_1
            writter_spk2[mix_key] = 16000, wav_2
            writter_mix[mix_key] = 16000, mixture
            writter_text_spk1.write(mix_key + " " + text_sources[key_1])
            writter_text_spk2.write(mix_key + " " + text_sources[key_2])
            

    writter_mix.close()
    writter_spk1.close()
    writter_spk2.close()
    writter_text_spk1.close()
    writter_text_spk2.close()


if __name__ == "__main__":
    main()
