import torch
from espnet.nets.pytorch_backend.transformer.encoder import Encoder as TransformerEncoder
from torch import nn as nn
from torch.nn import functional as F


class ResRNN(nn.Module):
    """
    Container module for a single RNN layer with linear output projection (for residual connection).

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state. The corresponding output should
                    have shape (batch, seq_len, hidden_size).
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
        residual: bool, whether to add layer-wise residual connection. Default is False.
    """

    def __init__(self, input_size, hidden_size, num_layers=1, rnn_type='LSTM', dropout=0, skip=False,
                 bidirectional=False):
        super(ResRNN, self).__init__()

        self.input_size = input_size
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1
        self.dropout = dropout

        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, num_layers, batch_first=True,
                                         bidirectional=bidirectional)
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)

    def forward(self, input):
        # input shape: batch, L, N
        output = input
        self.rnn.flatten_parameters()
        rnn_output, _ = self.rnn(output)
        proj_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(output.shape)
        output = F.dropout(proj_output, p=self.dropout)
        return output


class LocalRNN(nn.Module):
    """
        Local RNN
    """

    def __init__(self, input_size, hidden_size, dropout=0, bidirectional=True):
        super(LocalRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnn = ResRNN(input_size=input_size, hidden_size=hidden_size, rnn_type='LSTM', dropout=dropout,
                          bidirectional=bidirectional)
        self.norm = nn.GroupNorm(1, input_size, eps=torch.finfo(torch.float32).eps)

    def forward(self, input):
        # input shape: B, N, block, num_block
        batch_size, N, dim1, dim2 = input.shape
        output = input

        # intra-block RNN
        output = output.permute(0, 3, 2, 1).contiguous().view(batch_size * dim2, dim1, -1)  # B*dim2, dim1, N
        output = self.rnn(output)  # B*dim2, dim1, N
        output = output.view(batch_size, dim2, dim1, -1).permute(0, 3, 2, 1).contiguous()  # B, N, dim1, dim2
        output = input + self.norm(output)  # B, N, dim1, dim2

        return output


class GlobalRNN(nn.Module):
    """
        Global RNN
    """

    def __init__(self, input_size, hidden_size, dropout=0, bidirectional=True):
        super(GlobalRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.factor = int(bidirectional) + 1

        # DPRNN for 3D input (B, N, block, num_block)
        self.row_rnn = ResRNN(input_size=input_size, hidden_size=hidden_size, rnn_type='LSTM', dropout=dropout,
                              bidirectional=True)
        self.col_rnn = ResRNN(input_size=input_size, hidden_size=hidden_size, rnn_type='LSTM', dropout=dropout,
                              bidirectional=bidirectional)

        self.row_norm = nn.GroupNorm(1, input_size, eps=torch.finfo(torch.float32).eps)
        self.col_norm = nn.GroupNorm(1, input_size, eps=torch.finfo(torch.float32).eps)

    def forward(self, input):
        # input shape: B, N, block, num_block
        batch_size, N, dim1, dim2 = input.shape
        output = input

        # intra-block RNN
        row_input = output.permute(0, 3, 2, 1).contiguous().view(batch_size * dim2, dim1, -1)  # B*dim2, dim1, N
        row_output = self.row_rnn(row_input)  # B*dim2, dim1, N
        row_output = row_output.view(batch_size, dim2, dim1, -1).permute(0, 3, 2, 1).contiguous()  # B, N, dim1, dim2
        output = output + self.row_norm(row_output)  # B, N, dim1, dim2

        # inter-block RNN
        col_input = output.permute(0, 2, 3, 1).contiguous().view(batch_size * dim1, dim2, -1)  # B*dim1, dim2, N
        col_output = self.col_rnn(col_input)  # B*dim1, dim2, N
        col_output = col_output.view(batch_size, dim1, dim2, -1).permute(0, 3, 1, 2).contiguous()  # B, N, dim1, dim2
        output = output + self.col_norm(col_output)  # B, N, dim1, dim2

        return output


class GlobalATT(nn.Module):
    """
        Global Att
    """

    def __init__(self, input_size, hidden_size, dropout=0, bidirectional=True,
                 attention_dim=256, num_head=4, att_ff=1024):
        super(GlobalATT, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.factor = int(bidirectional) + 1

        # DPRNN for 3D input (B, N, block, num_block)
        self.row_rnn = ResRNN(input_size=input_size, hidden_size=hidden_size, rnn_type='LSTM', dropout=dropout,
                              bidirectional=True)

        self.col_att = TransformerEncoder(idim=input_size,
                                          input_layer='linear',
                                          attention_dim=attention_dim,
                                          linear_units=att_ff,
                                          attention_heads=num_head,
                                          num_blocks=1,
                                          dropout_rate=dropout,
                                          positional_dropout_rate=dropout,
                                          attention_dropout_rate=dropout,
                                          pos_enc_class=torch.nn.Identity,
                                          normalize_before=False)
        self.col_bn = nn.Linear(attention_dim, input_size)

        self.row_norm = nn.GroupNorm(1, input_size, eps=torch.finfo(torch.float32).eps)

    def forward(self, input):
        # input shape: B, N, block, num_block
        batch_size, N, dim1, dim2 = input.shape
        output = input

        # intra-block RNN
        row_input = output.permute(0, 3, 2, 1).contiguous().view(batch_size * dim2, dim1, -1)  # B*dim2, dim1, N
        row_output = self.row_rnn(row_input)  # B*dim2, dim1, N
        row_output = row_output.view(batch_size, dim2, dim1, N).permute(0, 3, 2, 1).contiguous()  # B, N, dim1, dim2
        output = output + self.row_norm(row_output)  # B, N, dim1, dim2

        # inter-block RNN
        col_input = output.permute(0, 2, 3, 1).contiguous().view(batch_size * dim1, dim2, N)  # B*dim1, dim2, N
        col_output, _ = self.col_att(col_input, None)  # B*dim1, dim2, N
        col_output = self.col_bn(col_output)
        col_output = col_output.view(batch_size, dim1, dim2, N).permute(0, 3, 1, 2).contiguous()  # B, N, dim1, dim2
        output = output + col_output
        return output