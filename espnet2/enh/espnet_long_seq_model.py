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
        self.stitching_loss = getattr(self.enh_model, 'stitching_loss', False)

    def compute_overlap_ratio(self, ref_1, ref_2):
        '''
        :param ref_1: tensor (block, T)
        :param ref_2: tensor (block, T)
        :return: tensor (block,)
        '''
        overlap = (ref_1 != 0) * (ref_2 != 0)
        ratio = overlap.float().mean(dim=-1)
        return ratio

    def compute_snr_ov(self, snrs, overlap_ratio):
        return_snrs = {}

        idx_0 = (overlap_ratio == 0)
        ov_0 = sum(idx_0 * snrs)
        idx_0_25 = (overlap_ratio < 0.25) * (overlap_ratio > 0)
        ov_0_25 = sum(idx_0_25 * snrs)
        idx_25_50 = (overlap_ratio > 0.25) * (overlap_ratio < 0.5)
        ov_25_50 = sum(idx_25_50 * snrs)
        idx_50_75 = (overlap_ratio > 0.5) * (overlap_ratio < 0.75)
        ov_50_75 = sum(idx_50_75 * snrs)
        idx_75_100 = (overlap_ratio > 0.75)
        ov_75_100 = sum(idx_75_100 * snrs)

        SNR_0 = ov_0 * 2
        cnt_0 = sum(idx_0)
        return_snrs['snr_0'] = (SNR_0 / cnt_0) if cnt_0 else 1e-8

        SNR_0_25 = ov_0_25
        cnt_0_25 = sum(idx_0_25)
        return_snrs['snr_0_25'] = (SNR_0_25 / cnt_0_25) if cnt_0_25 else 1e-8

        SNR_25_50 = ov_25_50
        cnt_25_50 = sum(idx_25_50)
        return_snrs['snr_25_50'] = (SNR_25_50 / cnt_25_50) if cnt_25_50 else 1e-8

        SNR_50_75 = ov_50_75
        cnt_50_75 = sum(idx_50_75)
        return_snrs['snr_50_75'] = (SNR_50_75 / cnt_50_75) if cnt_50_75 else 1e-8

        SNR_75_100 = ov_75_100
        cnt_75_100 = sum(idx_75_100)
        return_snrs['snr_75_100'] = (SNR_75_100 / cnt_75_100) if cnt_75_100 else 1e-8

        return return_snrs

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
        speech_ref = speech_ref[:, :, : speech_lengths.max()]  # B, 2, L
        speech_mix = speech_mix[:, : speech_lengths.max()]  # B, L
        speech_ref = speech_ref.unbind(dim=1)
        speech_ref = [self.enh_model.segmentation(ref) for ref in speech_ref]
        num_seg = speech_ref[0].shape[1]
        speech_ref = [ref.view(batch_size * num_seg, -1) for ref in speech_ref]
        speech_ref = torch.stack(speech_ref, dim=1)  # B*num_seg, 2, L'

        loss, speech_pre, mask_pre, out_lengths, perm = self._compute_loss(
            speech_mix,
            speech_lengths,
            speech_ref,
            noise_ref=noise_ref,
        )
        stats = dict(loss=loss.detach(), )

        if self.stitching_loss:
            stitch_loss = self._mask_stitching_loss(mask_pre, batch_size=batch_size)
            loss += stitch_loss
            stats['stitch_loss'] = stitch_loss.detach()

        if self.training:
            snr = None
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
            ov_ratio = self.compute_overlap_ratio(*speech_ref)
            si_snr_loss = self._permutation_loss_ov(
                speech_ref, speech_pre, self.snr_loss, overlap=ov_ratio
            ).detach()
            snrs = -si_snr_loss.detach()
            snrs_ov = self.compute_snr_ov(snrs, ov_ratio)
            snr = snrs.mean()
            stats.update(snrs_ov)

        stats['snr'] = snr
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
                        snr.append(10 * torch.log10(ref[i].pow(2).sum() + eps).view(1, ) - 10 * torch.log10(
                            noise[i].pow(2).sum() + eps).view(1, ))
                else:
                    snr.append(10 * torch.log10(ref[i].pow(2).sum() + eps).view(1, ) - 10 * torch.log10(
                        noise[i].pow(2).sum() + eps).view(1, ))
            snr = torch.cat(snr, 0)

        return -snr

    def _permutation_loss_ov(self, ref, inf, criterion, overlap=[-1]):
        """
        Args:
            ref (List[torch.Tensor]): [(batch, ...), ...]
            inf (List[torch.Tensor]): [(batch, ...), ...]
            criterion (function): Loss function
            overlap: List. Default is [-1] (do not consider overlap)
            perm: (batch)
        Returns:
            torch.Tensor: (batch)
        """
        num_spk = len(ref)

        def pair_loss(permutation):
            return sum(
                [criterion(ref[s], inf[t], overlap) for s, t in enumerate(permutation)]
            ) / len(permutation)

        losses = torch.stack([pair_loss(p) for p in permutations(range(num_spk))], dim=1)

        loss, perm = torch.min(losses, dim=1)

        return loss

    def _mask_stitching_loss(self, masks, batch_size=1):
        """
        :param masks: dict, {'spk1': [B*seg, T, F], 'spk2': [B*seg, T, F]}
        :return: stitching_loss
        """
        masks = [masks[key] for key in masks]

        _, T, F = masks[0].shape

        masks = [m.view(batch_size, -1, T, F) for m in masks]
        fronts = []
        ends = []
        for mask in masks:
            front, end = mask.split(T // 2, 2)
            front = front[:, 1:, :, :].view(-1, T // 2, F)
            end = end[:, :-1, :, :].view(-1, T // 2, F)
            fronts.append(front)
            ends.append(end)
        stitching_loss, _ = self._permutation_loss(fronts, ends, self.tf_mse_loss)
        return stitching_loss

    def _compute_loss(
            self,
            speech_mix,
            speech_lengths,
            speech_ref,
            noise_ref=None,
    ):
        """Compute loss according to self.loss_type.

        Args:
            speech_mix: (Batch, samples) or (Batch, samples, channels)
            speech_lengths: (Batch,), default None for chunk interator,
                            because the chunk-iterator does not have the
                            speech_lengths returned. see in
                            espnet2/iterators/chunk_iter_factory.py
            speech_ref: (Batch, num_speaker, samples)
                        or (Batch, num_speaker, samples, channels)
            noise_ref: (Batch, num_speaker, samples)
                        or (Batch, num_speaker, samples, channels)

        Returns:
            loss: (torch.Tensor) speech enhancement loss
            speech_pre: (List[torch.Tensor] or List[ComplexTensor]) enhanced speech or spectrum(s)
            mask_pre: (OrderedDict) estimated masks or None
            output_lengths: (Batch,)
            perm: () best permutation
        """
        eps = 1e-8
        # prepare reference speech and reference spectrum
        speech_ref = torch.unbind(speech_ref, dim=1)
        spectrum_ref = [self.enh_model.stft(sr)[0] for sr in speech_ref]

        # List[ComplexTensor(Batch, T, F)] or List[ComplexTensor(Batch, T, C, F)]
        spectrum_ref = [
            ComplexTensor(sr[..., 0], sr[..., 1]) for sr in spectrum_ref
        ]
        spectrum_mix = self.enh_model.stft(speech_mix)[0]
        spectrum_mix = ComplexTensor(spectrum_mix[..., 0], spectrum_mix[..., 1])

        # predict separated speech and masks
        spectrum_pre, tf_length, mask_pre = self.enh_model(
            speech_mix, speech_lengths
        )

        # compute TF masking loss
        if self.loss_type == "magnitude":
            # compute loss on magnitude spectrum
            magnitude_pre = [abs(ps + eps) for ps in spectrum_pre]
            if spectrum_ref[0].dim() > magnitude_pre[0].dim():
                # only select one channel as the reference
                magnitude_ref = [
                    abs(sr[..., self.ref_channel, :] + eps) for sr in spectrum_ref
                ]
            else:
                magnitude_ref = [abs(sr + eps) for sr in spectrum_ref]

            tf_loss, perm = self._permutation_loss(
                magnitude_ref, magnitude_pre, self.tf_mse_loss
            )
        elif self.loss_type.startswith("mask"):
            if self.loss_type == "mask_mse":
                loss_func = self.tf_mse_loss
            else:
                raise ValueError("Unsupported loss type: %s" % self.loss_type)

            assert mask_pre is not None
            mask_pre_ = [
                mask_pre["spk{}".format(spk + 1)] for spk in range(self.num_spk)
            ]

            # prepare ideal masks
            mask_ref = self._create_mask_label(
                spectrum_mix, spectrum_ref, mask_type=self.mask_type
            )

            # compute TF masking loss
            tf_loss, perm = self._permutation_loss(mask_ref, mask_pre_, loss_func)

            if "noise1" in mask_pre:
                if noise_ref is None:
                    raise ValueError(
                        "No noise reference for training!\n"
                        'Please specify "--use_noise_ref true" in run.sh'
                    )

                noise_ref = torch.unbind(noise_ref, dim=1)
                noise_spectrum_ref = [
                    self.enh_model.stft(nr)[0] for nr in noise_ref
                ]
                noise_spectrum_ref = [
                    ComplexTensor(nr[..., 0], nr[..., 1])
                    for nr in noise_spectrum_ref
                ]
                noise_mask_ref = self._create_mask_label(
                    spectrum_mix, noise_spectrum_ref, mask_type=self.mask_type
                )

                mask_noise_pre = [
                    mask_pre["noise{}".format(n + 1)]
                    for n in range(self.num_noise_type)
                ]
                tf_noise_loss, perm_n = self._permutation_loss(
                    noise_mask_ref, mask_noise_pre, loss_func
                )
                tf_loss = tf_loss + tf_noise_loss
        else:
            raise ValueError("Unsupported loss type: %s" % self.loss_type)

        loss = tf_loss
        return loss, spectrum_pre, mask_pre, tf_length, perm
