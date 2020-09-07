from collections import OrderedDict
from typing import Tuple

from espnet.nets.pytorch_backend.rnn.encoders import RNN
from espnet2.enh.abs_enh import AbsEnhancement
from espnet2.layers.stft import Stft
from espnet2.layers.utterance_mvn import UtteranceMVN
import torch
from torch_complex.tensor import ComplexTensor


class TFMaskingNetCTX(AbsEnhancement):
    """TF Masking Speech Separation Net."""

    def __init__(
            self,
            n_fft: int = 512,
            win_length: int = None,
            hop_length: int = 128,
            rnn_type: str = "blstm",
            layer: int = 3,
            unit: int = 512,
            dropout: float = 0.0,
            num_spk: int = 2,
            nonlinear: str = "sigmoid",
            enc_dim: int = 512,
            utt_mvn: bool = False,
            mask_type: str = "IRM",
            use_noise_mask: bool = False,
            loss_type: str = "mask_mse",
    ):
        super(TFMaskingNetCTX, self).__init__()

        self.num_spk = num_spk
        self.num_bin = n_fft // 2 + 1
        self.mask_type = mask_type
        self.loss_type = loss_type
        self.use_noise_mask = use_noise_mask
        if loss_type not in ("mask_mse", "magnitude", "magnitude_l1", "spectrum", "spectrum_l1"):
            raise ValueError("Unsupported loss type: %s" % loss_type)

        self.stft = Stft(n_fft=n_fft, win_length=win_length, hop_length=hop_length, )

        # self.utt_mvn = UtteranceMVN(norm_means=True, norm_vars=True)
        self.bottleneck_ctx = torch.nn.Linear(enc_dim, self.num_bin)
        self.rnn = RNN(
            idim=self.num_bin * (1 + self.num_spk),
            elayers=layer,
            cdim=unit,
            hdim=unit,
            dropout=dropout,
            typ=rnn_type,
        )

        self.linear = torch.nn.ModuleList(
            [torch.nn.Linear(unit, self.num_bin) for _ in range(self.num_spk + int(self.use_noise_mask))]
        )

        if nonlinear not in ("sigmoid", "relu", "tanh"):
            raise ValueError("Not supporting nonlinear={}".format(nonlinear))

        self.nonlinear = {
            "sigmoid": torch.nn.Sigmoid(),
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
        }[nonlinear]

    def forward(self, input: torch.Tensor, ctx: Tuple[torch.Tensor], ilens: torch.Tensor):
        """Forward.

        Args:
            input (torch.Tensor): mixed speech [Batch, sample]
            ilens (torch.Tensor): input lengths [Batch]

        Returns:
            separated (list[ComplexTensor]): [(B, T, F), ...]
            ilens (torch.Tensor): (B,)
            predcited masks: OrderedDict[
                'spk1': torch.Tensor(Batch, Frames, Channel, Freq),
                'spk2': torch.Tensor(Batch, Frames, Channel, Freq),
                ...
                'spkn': torch.Tensor(Batch, Frames, Channel, Freq),
            ]
        """

        # wave -> stft -> magnitude specturm
        input_spectrum, flens = self.stft(input, ilens)
        input_spectrum = ComplexTensor(input_spectrum[..., 0], input_spectrum[..., 1])
        input_magnitude = abs(input_spectrum)
        # input_phase = input_spectrum / (input_magnitude + 10e-12)

        # apply utt mvn
        # input_magnitude_mvn, fle = self.utt_mvn(input_magnitude, flens)

        cc = []
        for c in ctx:
            c = self.bottleneck_ctx(c)
            c = torch.nn.functional.interpolate(c.transpose(2, 1), input_magnitude.shape[1])
            c = c.transpose(2, 1)
            cc.append(c)
        # predict masks for each speaker
        x, flens, _ = self.rnn(torch.cat([input_magnitude, *cc], dim=2), flens)
        masks = []
        for linear in self.linear:
            y = linear(x)
            y = self.nonlinear(y)
            masks.append(y)

        if self.training and self.loss_type.startswith("mask"):
            predicted_spectrums = None
        else:
            # apply mask
            predicted_spectrums = [input_spectrum * m for m in masks[0:self.num_spk]]
        ret_masks = {}
        for i in range(self.num_spk):
            ret_masks[f'spk{i + 1}'] = masks[i]
        if self.use_noise_mask:
            ret_masks['noise1'] = masks[-1]

        return predicted_spectrums, flens, ret_masks

    def forward_rawwav(
            self, input: torch.Tensor, ilens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Output with waveforms.

        Args:
            input (torch.Tensor): mixed speech [Batch, sample]
            ilens (torch.Tensor): input lengths [Batch]

        Returns:
            predcited speech [Batch, num_speaker, sample]
            output lengths
            predcited masks: OrderedDict[
                'spk1': torch.Tensor(Batch, Frames, Channel, Freq),
                'spk2': torch.Tensor(Batch, Frames, Channel, Freq),
                ...
                'spkn': torch.Tensor(Batch, Frames, Channel, Freq),
            ]
        """

        # predict spectrum for each speaker
        predicted_spectrums, flens, masks = self.forward(input, ilens)

        if predicted_spectrums is None:
            predicted_wavs = None
        elif isinstance(predicted_spectrums, list):
            # multi-speaker input
            predicted_wavs = [
                self.stft.inverse(ps, ilens)[0] for ps in predicted_spectrums
            ]
        else:
            # single-speaker input
            predicted_wavs = self.stft.inverse(predicted_spectrums, ilens)[0]

        return predicted_wavs, ilens, masks
