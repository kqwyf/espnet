import torch
import torch.nn.functional as F
from torch_complex.tensor import ComplexTensor
import torch_complex.functional as cF
import numpy


def get_stitch(masks):
    masks = torch.stack(list(masks.values()))
    num_spk, num_segment, samples, fbin = masks.shape
    stitch_margin = samples // 2

    PERM = []
    for seg_idx in range(num_segment - 1):
        prev = masks[:, seg_idx, :, :]
        now = masks[:,seg_idx + 1, :, :]
        sim_matrix = torch.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                d = prev[j, -stitch_margin:, :] - now[i, :stitch_margin:, ]
                sim_matrix[i, j] = - torch.sum(d * d)

        sim0 = sim_matrix[0, 0] + sim_matrix[1, 1]
        sim1 = sim_matrix[0, 1] + sim_matrix[0, 1]
        if sim0 > sim1:
            perm = [0, 1]
        else:
            perm = [1, 0]
        PERM.append(perm)
    return PERM


def get_connect(spec, PERM):
    # wav: num_segment, num_spk, samples
    num_segment, segment, fbin = spec[0].shape
    num_spk = len(spec)
    stitch_margin = segment // 2
        # (ori_len - samples) // (num_segment - 1)
    state = 0
    N_M1 = [0]
    for i, item in enumerate(PERM):
        if item[0] == 1:
            state = 1 - state
        N_M1.append(state)

    res1 = []
    res2 = []

    for i in range(len(N_M1)):
        if N_M1[i] == 0:
            res1.append(spec[0][i])
            res2.append(spec[1][i])
        else:
            res1.append(spec[1][i])
            res2.append(spec[0][i])
    all_len = stitch_margin * (len(N_M1) - 1) + segment

    res_1 = ComplexTensor(numpy.zeros((all_len, fbin), dtype=numpy.complex64)).to(spec[0].device)
    res_2 = ComplexTensor(numpy.zeros((all_len, fbin), dtype=numpy.complex64)).to(spec[0].device)
    indicator = torch.zeros((all_len, 1)).to(spec[0].device)

    for i in range(len(N_M1)):
        st = stitch_margin * i
        en = st + segment
        res_1[st:en, :] += res1[i]
        res_2[st:en, :] += res2[i]
        indicator[st:en] += 1
    indicator[indicator == 0] = 1
    return cF.stack([res_1 / indicator, res_2 / indicator])


def get_connect_v2(wav, PERM, sep_net):
    # wav: num_segment, num_spk, samples
    wav = torch.stack(wav, dim=1)
    num_segment, num_spk, samples = wav.shape
    wav = wav.view(num_segment*num_spk, samples)

    encoder_weight = torch.eye(sep_net.window_size).type(wav.type()).unsqueeze(1).to(wav.device)
    enc_input = F.conv1d(wav.unsqueeze(1), encoder_weight, stride=sep_net.stride_size) # num_segment*num_spk, N, L
    enc_input = enc_input.view(num_segment, num_spk, sep_net.window_size, -1)

    num_segment, num_spk, N, L = enc_input.shape

    state = 0
    N_M1 = [0]
    for i, item in enumerate(PERM):
        if item[0] == 1:
            state = 1 - state
        N_M1.append(state)

    res1 = []
    res2 = []

    for i in range(len(N_M1)):
        if N_M1[i] == 0:
            res1.append(enc_input[i, 0, :, :])
            res2.append(enc_input[i, 1, :, :])
        else:
            res1.append(enc_input[i, 1, :, :])
            res2.append(enc_input[i, 0, :, :])
    res1 = torch.stack(res1).permute(1, 2, 0)
    res2 = torch.stack(res2).permute(1, 2, 0)
    res = sep_net.separator.merge_segment(torch.stack([res1, res2]), 0) / 2.0
    res1, res2 = F.conv_transpose1d(res, encoder_weight, stride=sep_net.stride_size).unbind(0)

    return res1, res2