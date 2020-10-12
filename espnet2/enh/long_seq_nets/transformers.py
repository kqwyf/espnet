from abc import ABC

import torch
from espnet.nets.pytorch_backend.transformer.embedding import ScaledPositionalEncoding, RelPositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder import Encoder as TransformerEncoder
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer as TransfomerEncoderLayer
from espnet.nets.pytorch_backend.conformer.encoder_layer import EncoderLayer as ConformerEncoderLayer
from espnet.nets.pytorch_backend.conformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.conformer.swish import Swish
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention, RelPositionMultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)

from torch import nn as nn


class PositionwiseFeedForwardRNN(torch.nn.Module):
    """Positionwise feed forward layer.

    :param int idim: input dimenstion
    :param int hidden_units: number of hidden units
    :param float dropout_rate: dropout rate

    """

    def __init__(self, idim, hidden_units, dropout_rate):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForwardRNN, self).__init__()
        # self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_1 = torch.nn.LSTM(idim, hidden_units, 1, batch_first=True, bidirectional=True)
        self.w_2 = torch.nn.Linear(2 * hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        """Forward funciton."""
        x, _ = self.w_1(x)

        return self.w_2(self.dropout(torch.relu(x)))


class ResTransformerLayer(nn.Module):
    def __init__(self, input_size=256, num_head=4, att_ff=2048, dropout=0, layer_type="transformer", gl='local'):
        super(ResTransformerLayer, self).__init__()
        self.input_size = input_size
        self.layer_type = layer_type
        if layer_type == 'transformer':
            self.trans_layer = TransfomerEncoderLayer(
                size=input_size,
                self_attn=MultiHeadedAttention(n_head=num_head, n_feat=input_size, dropout_rate=dropout),
                feed_forward=PositionwiseFeedForward(input_size, att_ff, dropout_rate=dropout),
                dropout_rate=dropout,
                normalize_before=True,
            )
        elif layer_type == 'conformer':
            attClass = RelPositionMultiHeadedAttention if gl == 'local' else MultiHeadedAttention
            self.trans_layer = ConformerEncoderLayer(
                size=input_size,
                self_attn=attClass(n_head=num_head, n_feat=input_size, dropout_rate=dropout),
                feed_forward=PositionwiseFeedForward(input_size, att_ff, dropout_rate=dropout, activation=Swish()),
                feed_forward_macaron=PositionwiseFeedForward(input_size, att_ff, dropout_rate=dropout,
                                                             activation=Swish()),
                conv_module=ConvolutionModule(input_size, kernel_size=7, activation=Swish()),
                dropout_rate=dropout,
                normalize_before=True,
            )
            pass

    def forward(self, input):
        # input shape: batch, L, N
        output, _ = self.trans_layer(input, None)

        return output


class LocalTransformer(nn.Module):
    """
        Global Att
    """

    def __init__(self, input_size, dropout=0, num_blocks=1, num_head=4, att_ff=1024, idx=1, layer_type="transformer"):
        super(LocalTransformer, self).__init__()

        self.input_size = input_size
        self.idx = idx
        if self.idx == 0:
            self.emb = ScaledPositionalEncoding(input_size, dropout_rate=dropout) \
                if layer_type == 'transformer' else RelPositionalEncoding(input_size, dropout_rate=dropout)

        # DPRNN for 3D input (B, N, block, num_block)
        self.row_transformer = ResTransformerLayer(input_size=input_size,
                                                   num_head=num_head, att_ff=att_ff, dropout=dropout,
                                                   layer_type=layer_type)
        self.norm = nn.GroupNorm(1, input_size, eps=torch.finfo(torch.float32).eps)

    def forward(self, input):
        # input shape: B, N, block, num_block
        if isinstance(input, tuple):
            input, pos_emb = input[0], input[1]
        else:
            input, pos_emb = input, None
        batch_size, N, dim1, dim2 = input.shape
        output = input

        # intra-block RNN
        row_input = output.permute(0, 3, 2, 1).contiguous().view(batch_size * dim2, dim1, N)  # B*dim2, dim1, N
        if self.idx == 0:
            row_input = self.emb(row_input)
            if isinstance(row_input, tuple):
                row_input, pos_emb = row_input[0], row_input[1]
            else:
                row_input, pos_emb = row_input, None
        if pos_emb is not None:
            row_input = (row_input, pos_emb)
        row_output = self.row_transformer(row_input)  # B*dim2, dim1, N

        if isinstance(row_output, tuple):
            row_output, pos_emb = row_output[0], row_output[1]
        else:
            row_output, pos_emb = row_output, None

        row_output = row_output.view(batch_size, dim2, dim1, N).permute(0, 3, 2, 1).contiguous()  # B, N, dim1, dim2

        row_output = input + self.norm(row_output)

        if pos_emb is not None:
            row_output = (row_output, pos_emb)

        return row_output


class GlobalTransformer(nn.Module):
    """
        Global Att
    """

    def __init__(self, input_size, dropout=0, num_blocks=1,
                 attention_dim=256, num_head=4, att_ff=1024, idx=1, layer_type="transformer"):
        super(GlobalTransformer, self).__init__()

        self.input_size = input_size
        self.idx = idx
        if self.idx == 0:
            self.emb = ScaledPositionalEncoding(input_size, dropout_rate=dropout) \
                if layer_type == 'transformer' else RelPositionalEncoding(input_size, dropout_rate=dropout)
        # DPRNN for 3D input (B, N, block, num_block)
        self.row_transformer = ResTransformerLayer(input_size=input_size,
                                                   num_head=num_head, att_ff=att_ff, dropout=dropout,
                                                   layer_type=layer_type, gl='local')

        self.col_transformer = ResTransformerLayer(input_size=input_size,
                                                   num_head=num_head, att_ff=att_ff, dropout=dropout,
                                                   layer_type=layer_type, gl='global')

        self.row_norm = nn.GroupNorm(1, input_size, eps=torch.finfo(torch.float32).eps)
        self.col_norm = nn.GroupNorm(1, input_size, eps=torch.finfo(torch.float32).eps)

    def forward(self, input):
        # input shape: B, N, block, num_block
        if isinstance(input, tuple):
            input, pos_emb = input[0], input[1]
        else:
            input, pos_emb = input, None
        batch_size, N, dim1, dim2 = input.shape
        output = input

        # intra-block RNN
        row_input = output.permute(0, 3, 2, 1).contiguous().view(batch_size * dim2, dim1, N)  # B*dim2, dim1, N
        if self.idx == 0:
            row_input = self.emb(row_input)
            if isinstance(row_input, tuple):
                row_input, pos_emb = row_input[0], row_input[1]
            else:
                row_input, pos_emb = row_input, None

        if pos_emb is not None:
            row_input = (row_input, pos_emb)

        row_output = self.row_transformer(row_input)  # B*dim2, dim1, N

        if isinstance(row_output, tuple):
            row_output, pos_emb = row_output[0], row_output[1]
        else:
            row_output, pos_emb = row_output, None

        row_output = row_output.view(batch_size, dim2, dim1, N).permute(0, 3, 2, 1).contiguous()  # B, N, dim1, dim2


        output = output + self.row_norm(row_output)  # B, N, dim1, dim2

        col_input = row_output.permute(0, 2, 3, 1).contiguous().view(batch_size * dim1, dim2, N)  # B*dim1, dim2, N
        col_output = self.col_transformer(col_input)  # B*dim1, dim2, N
        col_output = col_output.view(batch_size, dim1, dim2, N).permute(0, 3, 1, 2).contiguous()  # B, N, dim1, dim2
        output = output + self.col_norm(col_output)  # B, N, dim1, dim2

        if pos_emb is not None:
            output = (output, pos_emb)

        return output

# by chenda
