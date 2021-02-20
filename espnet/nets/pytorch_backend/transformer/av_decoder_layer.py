#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder self-attention layer definition."""

import torch
from torch import nn

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.decoder_layer import DecoderLayer


class DualDecoderLayer(nn.Module):
    """Dual-decoder layer module.

    Args:
        decoder1 (nn.Module): Decoder layer 1.
        decoder2 (nn.Module): Decoder layer 2.


    """

    def __init__(
        self,
        decoder1,
        decoder2,
    ):
        super(DualDecoderLayer, self).__init__()
        self.decoder1 = decoder1
        self.decoder2 = decoder2

    def forward(self, tgt, tgt_mask, memory1, memory1_mask, memory2, memory2_mask, cache=(None, None)):
        """Compute decoded features.

        Args:
            tgt (Tuple[torch.Tensor]): Input tensors Tuple[(#batch, maxlen_out, size)].
            tgt_mask (Tuple[torch.Tensor]): Masks for input tensors Tuple[(#batch, maxlen_out)].
            memory1 (torch.Tensor): Encoded memory, float32 (#batch, maxlen_in_1, size_1).
            memory1_mask (torch.Tensor): Encoded memory mask (#batch, maxlen_in_1).
            memory2 (torch.Tensor): Encoded memory, float32 (#batch, maxlen_in_2, size_2).
            memory2_mask (torch.Tensor): Encoded memory mask (#batch, maxlen_in_2).
            cache (Tuple[List[torch.Tensor]]): Tuple of list of cached tensors.
                Each tensor shape should be (#batch, maxlen_out - 1, size).

        Returns:
            torch.Tensor: Output tensor(#batch, maxlen_out, size).
            torch.Tensor: Mask for output tensor (#batch, maxlen_out).
            torch.Tensor: Encoded memory 1 (#batch, maxlen_in_1, size_1).
            torch.Tensor: Encoded memory 1 mask (#batch, maxlen_in_1).
            torch.Tensor: Encoded memory 2 (#batch, maxlen_in_2, size_2).
            torch.Tensor: Encoded memory 2 mask (#batch, maxlen_in_2).

        """
        x1, x2 = tgt
        tgt_mask1, tgt_mask2 = tgt_mask

        x1, tgt_mask1, memory1, memory1_mask = self.decoder1(x1, tgt_mask1, memory1, memory1_mask, cache[0])
        x2, tgt_mask2, memory2, memory2_mask = self.decoder2(x2, tgt_mask2, memory2, memory2_mask, cache[1])

        x = x1, x2
        tgt_mask = tgt_mask1, tgt_mask2

        return x, tgt_mask, memory1, memory1_mask, memory2, memory2_mask


class DualAttentionDecoderLayer(nn.Module):
    """Single decoder layer module with dual-attention mechanism.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        src_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)


    """

    def __init__(
        self,
        size_a,
        size_v,
        self_attn,
        src_attn_a,
        src_attn_v,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
    ):
        """Construct an DecoderLayer object."""
        super(DualAttentionDecoderLayer, self).__init__()
        self.size_a = size_a
        self.size_v = size_v
        self.self_attn = self_attn
        self.src_attn_a = src_attn_a
        self.src_attn_v = src_attn_v
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size_a)
        self.norm2 = LayerNorm(size_a)
        self.norm3 = LayerNorm(size_a)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear1 = nn.Linear(size_a + size_a, size_a)
        # We cannot assume that visual input has the same size as the audio features,
        # so we must concat and project the attention results at the src-attention.
        self.concat_linear2 = nn.Linear(size_a + size_a + size_v, size_a)

    def forward(self, tgt, tgt_mask, memory, memory_mask, visual, visual_mask, cache=None):
        """Compute decoded features.

        Args:
            tgt (torch.Tensor): Input tensor (#batch, maxlen_out, size).
            tgt_mask (torch.Tensor): Mask for input tensor (#batch, maxlen_out).
            memory (torch.Tensor): Encoded memory, float32 (#batch, maxlen_in_a, size_a).
            memory_mask (torch.Tensor): Encoded memory mask (#batch, maxlen_in_a).
            memory (torch.Tensor): Encoded visual, float32 (#batch, maxlen_in_v, size_v).
            memory_mask (torch.Tensor): Encoded visual mask (#batch, maxlen_in_v).
            cache (List[torch.Tensor]): List of cached tensors.
                Each tensor shape should be (#batch, maxlen_out - 1, size).

        Returns:
            torch.Tensor: Output tensor(#batch, maxlen_out, size).
            torch.Tensor: Mask for output tensor (#batch, maxlen_out).
            torch.Tensor: Encoded memory (#batch, maxlen_in_a, size_a).
            torch.Tensor: Encoded memory mask (#batch, maxlen_in_a).
            torch.Tensor: Encoded visual (#batch, maxlen_in_v, size_v).
            torch.Tensor: Encoded visual mask (#batch, maxlen_in_v).

        """
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == (
                tgt.shape[0],
                tgt.shape[1] - 1,
                self.size,
            ), f"{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = None
            if tgt_mask is not None:
                tgt_q_mask = tgt_mask[:, -1:, :]

        if self.concat_after:
            tgt_concat = torch.cat(
                (tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)), dim=-1
            )
            x = residual + self.concat_linear1(tgt_concat)
        else:
            x = residual + self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask))
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x_concat = torch.cat(
            (
                x,
                self.src_attn_a(x, memory, memory, memory_mask),
                self.src_attn_v(x, visual, visual, visual_mask)
            ), dim=-1
        )
        x = residual + self.concat_linear2(x_concat)
        if not self.normalize_before:
            x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, tgt_mask, memory, memory_mask, visual, visual_mask

