#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder self-attention layer definition."""

import torch

from torch import nn

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class AVEncoderLayer(nn.Module):
    """Audio-Visual Encoder layer module.
    Use dual attention mechanism to fuse audio and visual features.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
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
        size_x,
        size_k,
        num_k,
        self_attn,
        src_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
    ):
        """Construct an AVEncoderLayer object."""
        super(AVEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.proj_k = nn.Linear(size_k, size_x)
        self.norm1 = LayerNorm(size_x)
        self.norm2 = LayerNorm(size_x)
        self.norm3 = LayerNorm(size_x)
        self.dropout = nn.Dropout(dropout_rate)
        self.size_x = size_x
        self.size_k = size_k
        self.num_k = num_k
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size_x + size_x, size_x)
        self.concat_linear_src = nn.Linear(size_x + size_x * num_k, size_x)

    def forward(self, x, mask_x, ks, masks_k, cache=None):
        """Compute encoded features.

        Args:
            x (torch.Tensor): Input tensor (#batch, time1, size).
            mask_x (torch.Tensor): Mask tensor for the input (#batch, time1).
            ks (Tuple[torch.Tensor]): Input tensors for calculating keys and values of attention mechanism Tuple[(#batch, time2, size)].
            masks_k (Tuple[torch.Tensor]): Mask tensors for the input ks Tuple[(#batch, time2)].
            cache (torch.Tensor): Cache tensor of the input (#batch, time1 - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, size).
            torch.Tensor: Mask tensor (#batch, time1).
            Tuple[torch.Tensor]: Same as k Tuple[(#batch, time2, size)].
            Tuple[torch.Tensor]: Same as mask_k Tuple[(#batch, time2)].

        """
        residual = x
        raw_ks = ks
        ks = [self.proj_k(k) for k in ks]

        if self.normalize_before:
            x = self.norm1(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size_x)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask_x = None if mask_x is None else mask_x[:, -1:, :]

        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x_q, x, x, mask_x)), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(self.self_attn(x_q, x, x, mask_x))

        if not self.normalize_before:
            x = self.norm1(x[0])

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        attn_results = [self.src_attn(x, k, k, mask_k) for k, mask_k in zip(ks, masks_k)]
        x_concat = torch.cat((x, *attn_results), dim=-1)
        x = residual + self.concat_linear_src(x_concat)
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

        return x, mask_x, raw_ks, masks_k

