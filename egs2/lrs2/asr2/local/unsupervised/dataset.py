import torch
from torch.utils.data import Dataset
import skvideo.io
import librosa
import numpy as np
import random
import warnings
import torch

warnings.filterwarnings('ignore')

def scp_to_dic(scp_file):
    key_rxfile = {}
    with open(scp_file) as file:
        for line in file:
            (key, rxfile) = line.strip().split(' ')
            key_rxfile[key] = rxfile
    return key_rxfile


class AudioVisualDataset(Dataset):
    def __init__(self, video_scp, sec=4, rand_clip=True, fs=16000, vfs=25):
        """
        video_scp: scp path
        sec: clipped seconds
        """
        super(Dataset, self).__init__()
        self.videos = scp_to_dic(video_scp)
        self.keys = list(self.videos.keys())
        self.sec = sec
        self.rand_clip = rand_clip
        self.fs = fs
        self.vfs = vfs

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        video_file = self.videos[self.keys[item]]
        video = skvideo.io.vread(video_file)
#         r, g, b = video[..., 0], video[..., 1], video[..., 2]
#         video = (0.2989 * r + 0.5870 * g + 0.1140 * b) / 255

        audio, sr = librosa.load(video_file, sr=self.fs)

        # pad to length to sec if too short
        if len(audio) < int(self.fs * self.sec):
            tmp = np.zeros(int(self.fs * self.sec))
            tmp[0:len(audio)] += audio
            audio = tmp
        if len(video) < int(self.vfs * self.sec):
            tmp = np.zeros((int(self.vfs * self.sec), *video.shape[1:]))
            tmp[0:len(video)] += video
            video = tmp

        # clip data to fixed length
        start_idx = random.randint(0,len(video) - int(self.vfs * self.sec)) if self.rand_clip else 0

        video = video[start_idx:start_idx + int(self.vfs * self.sec)]
        audio = audio[start_idx * self.fs // self.vfs: start_idx * self.fs // self.vfs + int(self.fs * self.sec)]

        video = np.transpose(video, (3, 0, 1, 2)) / 255

        return audio.astype('float32'), video.astype('float32')