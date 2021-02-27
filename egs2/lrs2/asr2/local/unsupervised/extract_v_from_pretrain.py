#!/usr/bin/env python3

# Copyright 2020 Shanghai Jiao Tong University (Chenda Li)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import argparse
from distutils.util import strtobool
import logging
from collections import OrderedDict

import skvideo.io
import kaldiio
import numpy as np

from espnet.utils.cli_utils import get_commandline_args
from espnet.utils.cli_writers import file_writer_helper
from espnet2.utils.types import int_or_none
from espnet2.fileio.sound_scp import SoundScpWriter


def get_parser():
    parser = argparse.ArgumentParser(
        description="Extract visual feature from pretrained model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")

    parser.add_argument("--job-index", default=1, type=int, help="Job index, for choosing the GPU to be used.")
    parser.add_argument("--jobs-per-gpu", default=1, type=int, help="Number of jobs per GPU, for choosing the GPU to be used.")

    parser.add_argument("model_path", type=str, help="pretrained model path")
    parser.add_argument("input_video", type=str, help="input video scp")
    parser.add_argument("wspecifier", type=str, help="output file writer specifier")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    if 'CUDA_VISIBLE_DEVICES' not in os.environ or os.environ['CUDA_VISIBLE_DEVICES'] == '':
        os.environ['CUDA_VISIBLE_DEVICES'] = str((args.job_index - 1) // args.jobs_per_gpu)
    else:
        devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        os.environ['CUDA_VISIBLE_DEVICES'] = devices[(args.job_index - 1) // args.jobs_per_gpu]

    import torch
    from vgg_nets import VGGM_V

    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())

    net = VGGM_V(extract_emb=True)

    pretrained_dict = torch.load(args.model_path)

    v_state_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        if 'model_v' in k:
            name = k[8:]
            v_state_dict[name] = v
        else:
            continue
    net.load_state_dict(v_state_dict)
    net = net.cuda()
    net.eval()

    
    with torch.no_grad(), open(args.input_video) as input_scp, file_writer_helper(
        args.wspecifier,
        filetype='mat',
        compress=True,
        compression_method=2,
    ) as writer:
        for line in input_scp:
            line = line.strip()
            key, v_path = line.split()[0], line.split()[1]
            print(key, v_path)

            video = skvideo.io.vread(v_path)
            video = np.transpose(video, (3, 0, 1, 2)) / 255
            video = torch.tensor(video.astype('float32')).cuda()
            video = video.unsqueeze(0)
            emb = net(video)
            emb = emb.squeeze(0)
            writer[key] = emb.cpu().numpy()




if __name__ == "__main__":
    main()
