from functools import reduce
from itertools import combinations
from itertools import permutations
from itertools import product
import random
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
from torch_complex.tensor import ComplexTensor
from typeguard import check_argument_types

from espnet2.enh.abs_enh import AbsEnhancement
from espnet2.enh.nets.tasnet import TasNet
from espnet2.enh.nets.dprnn_raw import FaSNet_base as DPRNN
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.enh.espnet_model import ESPnetEnhancementModel

ALL_LOSS_TYPES = (
    # mse_loss(predicted_mask, target_label)
    "mask_mse",
    # mse_loss(enhanced_magnitude_spectrum, target_magnitude_spectrum)
    "magnitude",
    # mse_loss(enhanced_complex_spectrum, target_complex_spectrum)
    "spectrum",
    # si_snr(enhanced_waveform, target_waveform)
    "si_snr",
)


class ESPnetLongSeqModel(ESPnetEnhancementModel):
    """Speech enhancement or separation Frontend model"""

    def __init__(
            self,
            enh_model: Optional[AbsEnhancement],
    ):
        assert check_argument_types()

        torch.nn.Module.__init__(self)

        self.enh_model = enh_model
        self.num_spk = enh_model.num_spk
        self.num_noise_type = getattr(self.enh_model, "num_noise_type", 1)
        # get mask type for TF-domain models
        self.mask_type = getattr(self.enh_model, "mask_type", None)
        # get loss type for model training
        self.loss_type = getattr(self.enh_model, "loss_type", None)
        assert self.loss_type in ALL_LOSS_TYPES, self.loss_type
        # for multi-channel signal
        self.ref_channel = getattr(self.enh_model, "ref_channel", -1)

    @staticmethod
    def _create_mask_label(mix_spec, ref_spec, mask_type="IAM"):
        """Create mask label.

        :param mix_spec: ComplexTensor(B, T, F)
        :param ref_spec: [ComplexTensor(B, T, F), ...] or ComplexTensor(B, T, F)
        :param noise_spec: ComplexTensor(B, T, F)
        :return: [Tensor(B, T, F), ...] or [ComplexTensor(B, T, F), ...]
        """

        assert mask_type in [
            "IBM",
            "IRM",
            "IAM",
            "PSM",
            "NPSM",
            "PSM^2",
        ], f"mask type {mask_type} not supported"
        eps = 10e-8
        mask_label = []
        for r in ref_spec:
            mask = None
            if mask_type == "IBM":
                flags = [abs(r) >= abs(n) for n in ref_spec]
                mask = reduce(lambda x, y: x * y, flags)
                mask = mask.int()
            elif mask_type == "IRM":
                # TODO(Wangyou): need to fix this,
                #  as noise referecens are provided separately
                mask = abs(r) / (sum(([abs(n) for n in ref_spec])) + eps)
            elif mask_type == "IAM":
                mask = abs(r) / (abs(mix_spec) + eps)
                mask = mask.clamp(min=0, max=1)
            elif mask_type == "PSM" or mask_type == "NPSM":
                phase_r = r / (abs(r) + eps)
                phase_mix = mix_spec / (abs(mix_spec) + eps)
                # cos(a - b) = cos(a)*cos(b) + sin(a)*sin(b)
                cos_theta = (
                        phase_r.real * phase_mix.real + phase_r.imag * phase_mix.imag
                )
                mask = (abs(r) / (abs(mix_spec) + eps)) * cos_theta
                mask = (
                    mask.clamp(min=0, max=1)
                    if mask_label == "NPSM"
                    else mask.clamp(min=-1, max=1)
                )
            elif mask_type == "PSM^2":
                # This is for training beamforming masks
                phase_r = r / (abs(r) + eps)
                phase_mix = mix_spec / (abs(mix_spec) + eps)
                # cos(a - b) = cos(a)*cos(b) + sin(a)*sin(b)
                cos_theta = (
                        phase_r.real * phase_mix.real + phase_r.imag * phase_mix.imag
                )
                mask = (abs(r).pow(2) / (abs(mix_spec).pow(2) + eps)) * cos_theta
                mask = mask.clamp(min=-1, max=1)
            assert mask is not None, f"mask type {mask_type} not supported"
            mask_label.append(mask)
        return mask_label

    def forward(
            self,
            speech_mix: torch.Tensor,
            speech_mix_lengths: torch.Tensor = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech_mix: (Batch, samples) or (Batch, samples, channels)
            speech_ref: (Batch, num_speaker, samples)
                        or (Batch, num_speaker, samples, channels)
            speech_mix_lengths: (Batch,), default None for chunk interator,
                            because the chunk-iterator does not have the
                            speech_lengths returned. see in
                            espnet2/iterators/chunk_iter_factory.py
        """
        # clean speech signal of each speaker
        batch_size = speech_mix.shape[0]

        speech_ref = [
            kwargs["speech_ref{}".format(spk + 1)] for spk in range(self.num_spk)
        ]
        # (Batch, num_speaker, samples) or (Batch, num_speaker, samples, channels)
        speech_ref = torch.stack(speech_ref, dim=1)


        if "noise_ref1" in kwargs:
            # noise signal (optional, required when using
            # frontend models with beamformering)
            noise_ref = [
                kwargs["noise_ref{}".format(n + 1)] for n in range(self.num_noise_type)
            ]
            # (Batch, num_noise_type, samples) or
            # (Batch, num_noise_type, samples, channels)
            noise_ref = torch.stack(noise_ref, dim=1)
        else:
            noise_ref = None

        # dereverberated noisy signal
        # (optional, only used for frontend models with WPE)
        speech_lengths = (
            speech_mix_lengths
            if speech_mix_lengths is not None
            else torch.ones(batch_size).int() * speech_mix.shape[1]
        )
        assert speech_lengths.dim() == 1, speech_lengths.shape
        # Check that batch_size is unified
        assert speech_mix.shape[0] == speech_ref.shape[0] == speech_lengths.shape[0], (
            speech_mix.shape,
            speech_ref.shape,
            speech_lengths.shape,
        )

        # for data-parallel
        speech_ref = speech_ref[:, :, : speech_lengths.max()] # B, 2, L
        speech_mix = speech_mix[:, : speech_lengths.max()]    # B, L
        speech_ref = speech_ref.unbind(dim=1)
        speech_ref = [self.enh_model.segmentation(ref) for ref in speech_ref]
        num_seg = speech_ref[0].shape[1]
        speech_ref = [ref.view(batch_size*num_seg, -1) for ref in speech_ref]
        speech_ref = torch.stack(speech_ref, dim=1) # B*num_seg, 2, L'

        loss, speech_pre, mask_pre, out_lengths, perm = self._compute_loss(
            speech_mix,
            speech_lengths,
            speech_ref,
            dereverb_speech_ref=None,
            noise_ref=noise_ref,
        )

        # add stats for logging
        if self.loss_type != "snr":
            if self.training:
                si_snr = None
            else:
                b, _, L = speech_ref.shape
                ilens = torch.tensor([L] * b)
                speech_pre = [
                    self.enh_model.stft.inverse(ps, ilens)[0]
                    for ps in speech_pre
                ]
                speech_ref = torch.unbind(speech_ref, dim=1)
                if speech_ref[0].dim() == 3:
                    # For si_snr loss, only select one channel as the reference
                    speech_ref = [sr[..., self.ref_channel] for sr in speech_ref]
                # compute si-snr loss
                si_snr_loss, perm = self._permutation_loss(
                    speech_ref, speech_pre, self.snr_loss, perm=perm
                )
                si_snr = -si_snr_loss.detach()

            stats = dict(si_snr=si_snr, loss=loss.detach(), )
        else:
            stats = dict(si_snr=-loss.detach(), loss=loss.detach())

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight


    def snr_loss(self, ref, inf, overlap=[-1]):
        """
        :param ref: (block_num, samples)
        :param inf: (block_num, samples)
        :param overlap: (block_num)
        :return: (Batch)
        """

        eps = 10e-8

        noise = ref - inf

        if overlap[0] == -1:
            snr = 10 * torch.log10(ref.pow(2).sum(1) + eps) - 10 * torch.log10(noise.pow(2).sum(1) + eps)
        else:
            snr = []
            for i in range(ref.shape[0]):
                if overlap[i] == 0:
                    if ref[i].pow(2).sum() == 0:
                        snr.append(torch.zeros(1).to(ref.device))
                    else:
                        snr.append(10 * torch.log10(ref[i].pow(2).sum() + eps).view(1,) - 10 * torch.log10(noise[i].pow(2).sum() + eps).view(1,))
                else:
                    snr.append(10 * torch.log10(ref[i].pow(2).sum() + eps).view(1,) - 10 * torch.log10(noise[i].pow(2).sum() + eps).view(1,))
            snr = torch.cat(snr, 0)

        return -snr



