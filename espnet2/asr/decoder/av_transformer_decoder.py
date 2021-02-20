# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder definition."""
from typing import Any
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Dict

import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder_layer import DecoderLayer
from espnet.nets.pytorch_backend.transformer.av_decoder_layer import DualDecoderLayer
from espnet.nets.pytorch_backend.transformer.av_decoder_layer import DualAttentionDecoderLayer
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.scorer_interface import MultiSourcesBatchScorerInterface
from espnet2.asr.decoder.abs_av_decoder import AbsAVDecoder


class AV_BaseTransformerDecoder(AbsAVDecoder, MultiSourcesBatchScorerInterface):
    """Base class of Transfomer decoder module with audio-visual input.
    Use dual-attention from WLAS to fuse the information from audio and visual inputs.

    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        visual_input_size: dimension of attention
        dropout_rate: dropout rate
        positional_dropout_rate_audio: dropout rate for position encoding
        positional_dropout_rate_visual: dropout rate for position encoding
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        visual_input_size: int,
        dropout_rate: float = 0.1,
        positional_dropout_rate_audio: float = 0.1,
        positional_dropout_rate_visual: float = 0.1,
        input_layer_a: str = "embed",
        input_layer_v: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_decoder_output: bool = False,
    ):
        assert check_argument_types()
        super().__init__()
        attention_dim = encoder_output_size
        self.concat_decoder_output = concat_decoder_output
        if concat_decoder_output:
            decoder_output_size = attention_dim + attention_dim
        else:
            decoder_output_size = attention_dim

        if input_layer_a == "embed":
            self.embed_a = torch.nn.Sequential(
                torch.nn.Embedding(vocab_size, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate_audio),
            )
        elif input_layer_a == "linear":
            self.embed_a = torch.nn.Sequential(
                torch.nn.Linear(vocab_size, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate_audio),
            )
        else:
            raise ValueError(f"only 'embed' or 'linear' is supported: {input_layer_a}")

        if input_layer_v == "linear":
            self.embed_v = torch.nn.Sequential(
                torch.nn.Linear(visual_input_size, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate_visual),
            )
        else:
            raise ValueError(f"only 'linear' is supported: {input_layer_v}")

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm_a = LayerNorm(attention_dim)
            if self.concat_decoder_output:
                self.after_norm_v = LayerNorm(attention_dim)
        if use_output_layer:
            self.output_layer = torch.nn.Linear(decoder_output_size, vocab_size)
        else:
            self.output_layer = None

        # Must set by the inheritance
        self.decoders = None

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        additional: Dict,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in_a, feat_a)
            hlens: (batch)
            additional: additional input
            ys_in_pad:
                input token ids, int64 (batch, maxlen_out)
                if input_layer == "embed"
                input tensor (batch, maxlen_out, #mels) in the other cases
            ys_in_lens: (batch)
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """
        vs_pad, vlens = additional['visual'], additional['visual_length']
        tgt = ys_in_pad
        # tgt_mask: (B, 1, L)
        tgt_mask = (~make_pad_mask(ys_in_lens)[:, None, :]).to(tgt.device)
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m

        memory = hs_pad
        memory_mask = (~make_pad_mask(hlens))[:, None, :].to(memory.device)

        visual = vs_pad
        visual_mask = (~make_pad_mask(vlens))[:, None, :].to(visual.device)

        x = self.embed_a(tgt)
        v = self.embed_v(visual)

        if self.concat_decoder_output:
            x, tgt_mask = (x, x), (tgt_mask, tgt_mask)
        x, tgt_mask, memory, memory_mask, v, visual_mask = self.decoders(
            x, tgt_mask, memory, memory_mask, v, visual_mask
        )
        if self.normalize_before:
            if self.concat_decoder_output:
                assert torch.all(tgt_mask[0] == tgt_mask[1])
                tgt_mask = tgt_mask[0]
                x = torch.cat((self.after_norm_a(x[0]), self.after_norm_v(x[1])), dim=-1)
            else:
                x = self.after_norm_a(x)
        if self.output_layer is not None:
            x = self.output_layer(x)

        olens = tgt_mask.sum(1)
        return x, olens

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        visual: torch.Tensor,
        cache: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.

        Args:
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            memory: encoded memory, float32  (batch, maxlen_in_a, feat_a)
            visual: encoded visual, float32  (batch, maxlen_in_v, feat_v)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        x = self.embed_a(tgt)
        v = self.embed_v(visual)
        if cache is None:
            cache = [None] * len(self.decoders)
        new_cache = []
        for c, decoder in zip(cache, self.decoders):
            x, tgt_mask, memory, memory_mask, v, visual_mask = decoder(
                x, tgt_mask, memory, None, v, visual_mask, cache=c
            )
            new_cache.append(x)

        if self.normalize_before:
            if self.concat_decoder_output:
                y = torch.cat((self.after_norm_a(x[0][:, -1]), self.after_norm_v(x[1][:, -1])), dim=-1)
            else:
                y = self.after_norm(x[:, -1])
        else:
            if self.concat_decoder_output:
                y = torch.cat((x[0][:, -1], x[1][:, -1]), dim=-1)
            else:
                y = x[:, -1]
        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        return y, new_cache

    def score(self, ys, state, x):
        """Score."""
        memory, additional = x['encoder_output'], x['additional']
        visual, visual_length = additional['visual'], additional['visual_length']
        ys_mask = subsequent_mask(len(ys), device=memory.device).unsqueeze(0)
        logp, state = self.forward_one_step(
            ys.unsqueeze(0), ys_mask, memory.unsqueeze(0), visual, cache=state
        )
        return logp.squeeze(0), state

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: Dict
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (Dict):
                A dict containing at least the encoder feature that generates ys (n_batch, xlen, n_feat), with key 'encoder_output'.

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        memory, additional = xs['encoder_output'], xs['additional']
        visual, visual_length = additional['visual'], additional['visual_length']
        # merge states
        n_batch = len(ys)
        n_layers = len(self.decoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        # batch decoding
        ys_mask = subsequent_mask(ys.size(-1), device=memory.device).unsqueeze(0)
        logp, states = self.forward_one_step(ys, ys_mask, memory, visual, cache=batch_state)

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        return logp, state_list


class AV_TransformerDecoder(AV_BaseTransformerDecoder):
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        visual_input_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate_audio: float = 0.1,
        positional_dropout_rate_visual: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer_a: str = "embed",
        input_layer_v: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        decoder_layer_type: str = "dual_attention",
        concat_decoder_output: bool = False, # should be True when using 'dual_decoder'
    ):
        assert check_argument_types()

        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            visual_input_size=visual_input_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate_audio=positional_dropout_rate_audio,
            positional_dropout_rate_visual=positional_dropout_rate_visual,
            input_layer_a=input_layer_a,
            input_layer_v=input_layer_v,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
            concat_decoder_output=concat_decoder_output,
        )

        attention_dim = encoder_output_size

        if decoder_layer_type == "dual_decoder":
            self.decoders = repeat(
                num_blocks,
                lambda lnum: DualDecoderLayer(
                    DecoderLayer(
                        attention_dim,
                        MultiHeadedAttention(
                            attention_heads, attention_dim, self_attention_dropout_rate
                        ),
                        MultiHeadedAttention(
                            attention_heads, attention_dim, src_attention_dropout_rate
                        ),
                        PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                        dropout_rate,
                        normalize_before,
                        concat_after,
                    ),
                    DecoderLayer(
                        attention_dim,
                        MultiHeadedAttention(
                            attention_heads, attention_dim, self_attention_dropout_rate
                        ),
                        MultiHeadedAttention(
                            attention_heads, attention_dim, src_attention_dropout_rate
                        ),
                        PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                        dropout_rate,
                        normalize_before,
                        concat_after,
                    ),
                ),
            )
        elif decoder_layer_type == "dual_attention":
            self.decoders = repeat(
                num_blocks,
                lambda lnum: DualAttentionDecoderLayer(
                    attention_dim,
                    attention_dim,
                    MultiHeadedAttention(
                        attention_heads, attention_dim, self_attention_dropout_rate
                    ),
                    MultiHeadedAttention(
                        attention_heads, attention_dim, src_attention_dropout_rate
                    ),
                    MultiHeadedAttention(
                        attention_heads, attention_dim, src_attention_dropout_rate
                    ),
                    PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ),
            )
        else:
            raise NotImplementedError("decoder_layer_type only supports dual_attention or dual_decoder.")

