import torch
from espnet2.asr.encoder.rnn_encoder import RNNEncoder
from espnet2.asr.encoder.vgg_rnn_encoder import VGGRNNEncoder
from espnet2.asr.frontend.default import DefaultFrontend


class CTXPredictor(torch.nn.Module):
    def __init__(self, fs=8000, n_fft=256, hop_length=64, me_layer=2, se_layer=1, hidden=512, enc_dim=256, num_spk=2,
                 dropout=0.0):
        super(CTXPredictor, self).__init__()
        self.hidden = hidden
        self.num_spk = num_spk
        self.front_end = DefaultFrontend(fs=fs, n_fft=n_fft, hop_length=hop_length)
        self.mix_encoder = VGGRNNEncoder(input_size=80, rnn_type='lstm', bidirectional=True, num_layers=me_layer,
                                         hidden_size=hidden, output_size=hidden * num_spk,
                                         dropout=dropout)
        self.sd_encoder = RNNEncoder(input_size=hidden, num_layers=se_layer, hidden_size=hidden, output_size=enc_dim,
                                     subsample=None, dropout=dropout)

    def forward(self, input: torch.Tensor, ilens: torch.Tensor):
        feats, f_len = self.front_end(input, ilens)
        enc_mix, h_len, _ = self.mix_encoder(feats, f_len)
        enc_list = enc_mix.split(self.hidden, dim=2)
        spk_enc = []
        for enc in enc_list[0:self.num_spk]:
            e, h_l, _ = self.sd_encoder(enc, h_len)
            spk_enc.append(e)

        return spk_enc, h_len
