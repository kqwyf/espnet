# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""
from typing import Optional
from typing import Tuple
from typing import Dict
from typing import List

import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.av_encoder_layer import AVEncoderLayer
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling2
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling6
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling8
from espnet2.layers.convolutions import Conv1dRes
from espnet2.asr.encoder.abs_av_encoder import AbsAVEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder


class AV_TransformerEncoderMixAtt(AbsAVEncoder, TransformerEncoder, torch.nn.Module):
    """Transformer encoder module with visual input.

    Args:
        input_size: input dim
        output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate in attention
        positional_dropout_rate: dropout rate after adding positional encoding
        input_layer: input layer type
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
        positionwise_layer_type: linear of conv1d
        positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
        padding_idx: padding_idx for input_layer=embed
    """

    def __init__(
        self,
        input_size: int,
        input_size_v: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks_sd: int = 4,
        num_blocks_rec: int = 8,
        num_blocks_vis: int = 2,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        input_layer_v: Optional[str] = "raw",
        subsample_layer: Optional[str] = None,
        visual_transformer_input_layer: Optional[str] = "linear",
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        padding_idx: int = -1,
        num_spkrs: int = 2,
        encoder_layer_type: str = "default",
    ):
        assert check_argument_types()
        """Construct an Encoder object."""
        super(AV_TransformerEncoderMixAtt, self).__init__(
            input_size=input_size,
            output_size=output_size,
            attention_heads=attention_heads,
            linear_units=linear_units,
            num_blocks=num_blocks_rec,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            attention_dropout_rate=self_attention_dropout_rate,
            input_layer=input_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
            concat_after=concat_after,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            padding_idx=padding_idx,
        )
        self._output_size = output_size
        self._output_size_v = input_size_v
        self.num_spkrs = num_spkrs
        self.encoder_layer_type = encoder_layer_type

        hidden_visual_size = output_size
        if input_layer_v == "linear":
            self.embed_v = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer_v == "conv1d":
            self.embed_v = Conv1dRes(
                input_size_v,
                output_size,
                dropout_rate=dropout_rate,
                pos_enc=pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer_v is None:
            self.embed_v = torch.nn.Sequential(
                pos_enc_class(output_size, positional_dropout_rate)
            )
        elif input_layer_v == "raw":
            self.embed_v = None
            hidden_visual_size = input_size_v
        elif input_layer_v == "transformer":
            self.embed_v = TransformerEncoder(
                input_size=input_size_v,
                output_size=output_size,
                attention_heads=attention_heads,
                linear_units=linear_units,
                num_blocks=num_blocks_vis,
                dropout_rate=dropout_rate,
                positional_dropout_rate=positional_dropout_rate,
                attention_dropout_rate=self_attention_dropout_rate,
                input_layer=visual_transformer_input_layer,
                pos_enc_class=pos_enc_class,
                normalize_before=normalize_before,
                concat_after=concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                padding_idx=padding_idx,
            )
        else:
            raise ValueError("unknown input_layer_v: " + input_layer_v)

        if subsample_layer == "conv2d":
            self.subsample = Conv2dSubsampling(output_size, output_size, dropout_rate)
        elif subsample_layer == "conv2d2":
            self.subsample = Conv2dSubsampling2(output_size, output_size, dropout_rate)
        elif subsample_layer == "conv2d6":
            self.subsample = Conv2dSubsampling6(output_size, output_size, dropout_rate)
        elif subsample_layer == "conv2d8":
            self.subsample = Conv2dSubsampling8(output_size, output_size, dropout_rate)
        elif subsample_layer is None:
            self.subsample = None
        else:
            raise ValueError("unknown subsample_layer: " + subsample_layer)

        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        if encoder_layer_type == "default":
            encoder_layer_lambda = lambda lnum: EncoderLayer(
                output_size,
                MultiHeadedAttention(
                    attention_heads, output_size, self_attention_dropout_rate
                ),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            )
        elif encoder_layer_type == "query_audio":
            raise NotImplementedError("query_audio is not supported now.")
            encoder_layer_lambda = lambda lnum: AVEncoderLayer(
                hidden_size_v,
                output_size,
                1,
                MultiHeadedAttention(
                    attention_heads, input_size_v, self_attention_dropout_rate
                ),
                MultiHeadedAttention(
                    attention_heads, input_size_v, src_attention_dropout_rate
                ),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            )
        elif encoder_layer_type == "query_visual":
            encoder_layer_lambda = lambda lnum: AVEncoderLayer(
                output_size,
                hidden_visual_size,
                num_spkrs,
                MultiHeadedAttention(
                    attention_heads, output_size, self_attention_dropout_rate
                ),
                MultiHeadedAttention(
                    attention_heads, output_size, src_attention_dropout_rate
                ),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            )
        else:
            raise NotImplementedError("Support only default, query_audio or query_visual.")
        self.encoders_sd = torch.nn.ModuleList(
            [
                repeat(
                    num_blocks_sd,
                    encoder_layer_lambda,
                )
                for i in range(num_spkrs)
            ]
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size

    def output_size_v(self) -> int:
        return self._output_size_v

    def _repaint_visual_paddings(self, visuals: List[torch.Tensor], visual_lengths: List[torch.LongTensor]) -> List[torch.Tensor]:
        """
        Repad the padded value by 0.
        """
        assert len(visuals[0].shape) == 3 # (B, L, D)
        batch_size, _, dim = visuals[0].shape
        visuals_ = []
        for vs, v_lens in zip(visuals, visual_lengths):
            l = v_lens.max()
            vs_ = torch.zeros((batch_size, l, dim), device=vs.device)
            for i in range(len(vs)):
                vs_[i, :v_lens[i]] += vs[i, :v_lens[i]]
            visuals_.append(vs_)
        return visuals_

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        additional: Dict,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        visuals = [additional['spkr_list'][i]['visual'] for i in range(self.num_spkrs)]
        visual_lengths = [additional['spkr_list'][i]['visual_length'] for i in range(self.num_spkrs)]
        visuals = self._repaint_visual_paddings(visuals, visual_lengths)
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        visual_masks = [(~make_pad_mask(vlens)[:, None, :]).to(xs_pad.device) for vlens in visual_lengths]

        if (
            isinstance(self.embed, Conv2dSubsampling)
            or isinstance(self.embed, Conv2dSubsampling2)
            or isinstance(self.embed, Conv2dSubsampling6)
            or isinstance(self.embed, Conv2dSubsampling8)
        ):
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)

        if self.embed_v is not None:
            if isinstance(self.embed_v, TransformerEncoder):
                visuals, visual_lengths, _ = list(zip(*[self.embed_v(v, vlens) for v, vlens in zip(visuals, visual_lengths)]))
                visual_masks = [None] * len(visuals) # old masks should be dropped
            elif (
                isinstance(self.embed_v, Conv2dSubsampling)
                or isinstance(self.embed_v, Conv2dSubsampling2)
                or isinstance(self.embed_v, Conv2dSubsampling6)
                or isinstance(self.embed_v, Conv2dSubsampling8)
            ):
                visuals, visual_masks = list(zip(*[self.embed_v(v, vmask) for v, vmask in zip(visuals, visual_masks)]))
                visual_lengths = [None] * len(visuals)
            else:
                visuals = [self.embed_v(v) for v in visuals]
                visual_masks = [None] * len(visuals) # old masks should be dropped
        xs_sd, masks_sd = [None] * self.num_spkrs, [None] * self.num_spkrs

        for ns in range(self.num_spkrs):
            if self.encoder_layer_type == "default":
                xs_sd[ns], masks_sd[ns] = self.encoders_sd[ns](xs_pad, masks)
            elif self.encoder_layer_type == "query_audio":
                raise NotImplementedError("query_audio is not supported now.")
            elif self.encoder_layer_type == "query_visual":
                xs_sd[ns], masks_sd[ns], _, _ = self.encoders_sd[ns](xs_pad, masks, visuals, visual_masks)
            else:
                raise NotImplementedError("Support only default, query_audio or query_visual.")
            if self.subsample is not None:
                xs_sd[ns], masks_sd[ns] = self.subsample(xs_sd[ns], masks_sd[ns])
            xs_sd[ns], masks_sd[ns] = self.encoders(xs_sd[ns], masks_sd[ns])

            if self.normalize_before:
                xs_sd[ns] = self.after_norm(xs_sd[ns])

        olens_sd = [m.squeeze(1).sum(1) for m in masks_sd]
        encoder_additional_out = {
            "spkr_list": [
                {
                    "visual": v,
                    "visual_length": vlen,
                    "visual_mask": vmask
                }
                for v, vlen, vmask in zip(visuals, visual_lengths, visual_masks)
            ]
        }
        return xs_sd, olens_sd, None, encoder_additional_out
