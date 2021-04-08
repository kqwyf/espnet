#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convolution layer definition."""

from typing import Sequence
import torch

from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding


class Conv1dUpsampling(torch.nn.Module):
    """Convolutional 1D upsampling (to 4 times length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim):
        """Construct an Conv1dUpsampling object."""
        super(Conv1dUpsampling, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(idim, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose1d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Linear(odim, odim)

    def forward(self, x, ilens=None):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time * 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time * 4.

        """
        x = x.transpose(2, 1)
        x = self.conv(x)
        x = x.transpose(2, 1)
        x = self.out(x)
        if ilens is None:
            return x
        else:
            olens = (ilens - 1) * 2 + 3
            return x, olens


class Conv1dChannelUpsampling(torch.nn.Module):
    """Convolutional 1D upsampling (to 4 times length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim):
        """Construct an Conv1dUpsampling object."""
        super(Conv1dChannelUpsampling, self).__init__()
        self.odim = odim
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(idim, odim * 2, 3, padding=1),
            torch.nn.ReLU(),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(odim, odim * 2, 3, padding=1),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Linear(odim, odim)

    def forward(self, x, ilens=None):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time * 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time * 4.

        """
        batch_size = x.shape[0]
        x = self.conv1(x.transpose(2, 1)).transpose(2, 1).contiguous().view(batch_size, -1, self.odim)
        x = self.conv2(x.transpose(2, 1)).transpose(2, 1).contiguous().view(batch_size, -1, self.odim)
        x = self.out(x)
        if ilens is None:
            return x
        else:
            olens = ilens * 2
            return x, olens


class Conv1dRes(torch.nn.Module):
    """Convolutional 1D Resnet.

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, num_layers=2, dropout_rate=0.1, pos_enc=None):
        """Construct an Conv1dRes object."""
        super(Conv1dRes, self).__init__()

        convs = [
            torch.nn.Sequential(
                torch.nn.Conv1d(idim, odim, 3, padding=1),
                torch.nn.ReLU()
            )
        ]
        for _ in range(1, num_layers):
            convs.append(
                torch.nn.Sequential(
                    torch.nn.Conv1d(odim, odim, 3, padding=1),
                    torch.nn.ReLU()
                )
            )
        self.convs = torch.nn.ModuleList(convs)

        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim, odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, ilens=None):
        """Convolute x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time, odim),
            torch.Tensor: Subsampled mask (#batch, 1, time),

        """
        x = x.transpose(2, 1)
        x = self.convs[0](x)
        for layer in self.convs[1:]:
            x = x + layer(x)
        x = x.transpose(2, 1)
        x = x + self.out(x)

        if ilens is None:
            return x
        else:
            return x, ilens

