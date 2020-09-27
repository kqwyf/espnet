from collections import OrderedDict
from typing import Tuple

from espnet.nets.pytorch_backend.rnn.encoders import RNN
from espnet2.enh.abs_enh import AbsEnhancement
from espnet2.asr.encoder.vgg_rnn_encoder import VGGRNNEncoder
from espnet2.asr.encoder.rnn_encoder import RNNEncoder
from espnet2.layers.stft import Stft
from espnet2.layers.utterance_mvn import UtteranceMVN
import torch
from torch_complex.tensor import ComplexTensor
from espnet2.asr.frontend.default import DefaultFrontend
import torch.nn.functional as F


class CTXPredictor(torch.nn.Module):
    def __init__(self, fs=8000, n_fft=256, hop_length=64, me_layer=2, se_layer=1, hidden=512, enc_dim=256, num_spk=2,
                 noise_enc=True, ctx_dropout=0.0):
        super(CTXPredictor, self).__init__()
        self.hidden = hidden
        self.noise_enc = noise_enc
        self.num_spk = num_spk
        self.front_end = DefaultFrontend(fs=fs, n_fft=n_fft, hop_length=hop_length)
        self.mix_encoder = VGGRNNEncoder(input_size=80, rnn_type='lstm', bidirectional=True, num_layers=me_layer,
                                         hidden_size=hidden, output_size=hidden * (num_spk + int(noise_enc)),
                                         dropout=ctx_dropout)
        self.sd_encoder = RNNEncoder(input_size=hidden, num_layers=se_layer, hidden_size=hidden, output_size=enc_dim,
                                     subsample=None, dropout=ctx_dropout)
        if noise_enc:
            self.noise_encoder = RNNEncoder(input_size=hidden, num_layers=se_layer, hidden_size=hidden,
                                            output_size=enc_dim,
                                            dropout=ctx_dropout,
                                            subsample=None)

    def forward(self, input: torch.Tensor, ilens: torch.Tensor):
        feats, f_len = self.front_end(input, ilens)
        enc_mix, h_len, _ = self.mix_encoder(feats, f_len)
        enc_list = enc_mix.split(self.hidden, dim=2)
        spk_enc = []
        for enc in enc_list[0:self.num_spk]:
            e, h_l, _ = self.sd_encoder(enc, h_len)
            spk_enc.append(e)
        if self.noise_enc:
            n_enc, _, _ = self.noise_encoder(enc_list[-1], h_len)
        else:
            n_enc = None

        return spk_enc, n_enc, h_len


class TFMaskingNet_Joint_CTX(AbsEnhancement):
    """TF Masking Speech Separation Net."""

    def __init__(
            self,
            n_fft: int = 512,
            win_length: int = None,
            hop_length: int = 128,
            fs: int = 8000,
            rnn_type: str = "blstm",
            layer: int = 3,
            unit: int = 512,
            dropout: float = 0.0,
            num_spk: int = 2,
            nonlinear: str = "sigmoid",
            utt_mvn: bool = False,
            mask_type: str = "IRM",
            use_noise_mask: bool = False,
            loss_type: str = "mask_mse",
            ctx_n_fft: int = 256,
            ctx_n_hop: int = 64,
            me_layer: int = 2,
            se_layer: int = 1,
            ctx_e_hidden: int = 512,
            ctx_dropout: float = 0.1,
            enc_dim: int = 256,
    ):
        super(TFMaskingNet_Joint_CTX, self).__init__()
        self.num_spk = num_spk
        self.fs = fs
        self.num_bin = n_fft // 2 + 1
        self.mask_type = mask_type
        self.loss_type = loss_type
        self.use_noise_mask = use_noise_mask
        if loss_type not in ("mask_mse", "magnitude", "magnitude_l1", "spectrum", "spectrum_l1"):
            raise ValueError("Unsupported loss type: %s" % loss_type)

        self.ctx_pre = CTXPredictor(fs=fs, n_fft=ctx_n_fft, hop_length=ctx_n_hop,
                                    me_layer=me_layer, se_layer=se_layer, hidden=ctx_e_hidden,
                                    enc_dim=enc_dim, num_spk=num_spk, noise_enc=use_noise_mask, ctx_dropout=ctx_dropout)
        self.bottleneck_ctx = torch.nn.Linear(enc_dim, self.num_bin)
        self.bottleneck_noise = torch.nn.Linear(enc_dim, self.num_bin) if use_noise_mask else None

        self.stft = Stft(n_fft=n_fft, win_length=win_length, hop_length=hop_length, )

        if utt_mvn:
            self.utt_mvn = UtteranceMVN(norm_means=True, norm_vars=True)

        else:
            self.utt_mvn = None

        self.rnn = RNN(
            idim=self.num_bin * (1 + num_spk + int(use_noise_mask)),
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

    def forward(self, input: torch.Tensor, ilens: torch.Tensor):
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

        spks_ctx_pre, n_enc, h_len = self.ctx_pre(input, ilens)
        spks_ctx = [self.bottleneck_ctx(c) for c in spks_ctx_pre]
        spks_ctx = [F.interpolate(c.transpose(2, 1), input_magnitude.shape[1]) for c in spks_ctx]
        spks_ctx = [c.transpose(2, 1) for c in spks_ctx]
        if self.bottleneck_noise:
            n_enc = self.bottleneck_noise(n_enc)
            n_enc = F.interpolate(n_enc.transpose(2, 1), input_magnitude.shape[1])
            n_enc = n_enc.transpose(2, 1)
        encs = [*spks_ctx, n_enc] if n_enc is not None else spks_ctx

        # apply utt mvn

        # predict masks for each speaker
        x, flens, _ = self.rnn(torch.cat([input_magnitude, *encs], dim=2), flens)
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

        return predicted_spectrums, spks_ctx_pre, flens, ret_masks

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
        predicted_spectrums, spks_ctx_pre, flens, masks = self.forward(input, ilens)

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


if __name__ == '__main__':
    ctx_predictor = TFMaskingNet_Joint_CTX(use_noise_mask=False)
    wav = torch.randn([1, 5 * 8000])
    len = torch.tensor([5 * 8000])

    out = ctx_predictor(wav, len)
    print(out)
