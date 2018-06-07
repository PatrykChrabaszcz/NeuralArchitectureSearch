from src.deep_learning.pytorch.models.cnn import SeparateChannelCNN, AllChannelsCNN
from src.deep_learning.pytorch.models.cnn import SplitConv
from src.deep_learning.pytorch.models.rnn import RNN, RNNLockedDropout
from src.deep_learning.pytorch.models.model_base import RnnBase
from src.deep_learning.pytorch.models.rnn.rnn_batch_norm import RNNBatchNorm
from torch.nn import BatchNorm1d
from torch.autograd import Variable
from torch import cat, from_numpy
import torch.nn as nn
import logging
import numpy as np
from torch import transpose
from torch.nn import functional as f

logger = logging.getLogger(__name__)


class ConvRNN(RnnBase):
    @staticmethod
    def add_arguments(parser):
        RnnBase.add_arguments(parser)
        parser.add_argument("features_per_channel", type=int, default=25,
                            help="Number of filters in the first cnn block.")
        parser.add_argument("time_kernel_size", type=int, default=40,
                            help="Width in time dimension of the kernels in the first cnn block.")

        parser.add_argument("merged_channels", type=int, default=40,
                            help="Number of filters in the second cnn block.")

        parser.add_argument("pooling_size", type=int, default=75,
                            help="Pooling size.")
        parser.add_argument("pooling_stride", type=int, default=15,
                            help="Pooling stride.")

        parser.add_argument("cnn_batch_norm", type=int, default=0,
                            help="Whether or not to use batch norm")
        return parser

    def __init__(self, features_per_channel, merged_channels, time_kernel_size, pooling_size, pooling_stride,
                 cnn_batch_norm, **kwargs):

        super().__init__(**kwargs)
        self.input_dropout_layer = RNNLockedDropout(self.dropout_i, use_mc_dropout=self.use_mc_dropout)

        self.split_cnn = SplitConv(in_size=self.input_size, middle_size=features_per_channel, out_size=merged_channels,
                                   time_kernel_size=time_kernel_size, input_in_rnn_format=False)

        self.pool = nn.AvgPool1d(kernel_size=pooling_size, stride=pooling_stride)

        if cnn_batch_norm:
            self.bn = BatchNorm1d(merged_channels)
        self.cnn_batch_norm = cnn_batch_norm

        cell = RnnBase.cell_mapper[self.rnn_cell_type]
        self.rnn = RNN(cell=cell, in_size=merged_channels, hidden_size=self.rnn_hidden_size,
                       num_layers=self.rnn_num_layers, dropout_f=self.dropout_f, dropout_h=self.dropout_h,
                       rnn_normalization=self.rnn_normalization, dilation=self.rnn_dilation, skip_mode=self.skip_mode,
                       skip_first=self.skip_first, skip_last=self.skip_last, use_mc_dropout=self.use_mc_dropout)

        self.fc = nn.Linear(in_features=self.rnn_hidden_size, out_features=self.output_size, bias=True)

    def forward(self, x, hidden, context):
        batch_size = x.size(0)
        time_size = x.size(1)

        if self.use_context:
            # Repeat context for each time-step
            context = cat([context] * time_size, dim=1)
            x = cat([x, context], dim=2)
            assert list(x.size()) == [batch_size, time_size, self.input_size + self.context_size]

        # Dropout on the input features
        x = self.input_dropout_layer(x)

        x = transpose(x, 1, 2)
        # Conv layers
        x = self.split_cnn(x)
        x = f.relu(x)
        if self.cnn_batch_norm:
            x = self.bn(x)

        x = self.pool(x)
        x = transpose(x, 1, 2)

        # Rnn with extended cell, dont take output hidden state because here we have different evaluation type
        # (Because we have pooling layer with stride)
        lstm_out, _ = self.rnn(x, hidden)
        lstm_out = lstm_out.contiguous()

        fc_out = self.fc(lstm_out.view(-1, lstm_out.size(2)))
        fc_out = fc_out.view(batch_size, -1, fc_out.size(1))

        return fc_out[:, -1:, :], hidden

    def offset_size(self, sequence_size):
        # Forward dummy vector and find out what is the output shape

        v = np.zeros((1, sequence_size, self.input_size), np.float32)
        v = Variable(from_numpy(v))

        c = None

        s = self.import_state([self.initial_state()])

        if next(self.parameters()).is_cuda:
            v = v.cuda()
            #c = c.cuda()
            s = [_s.cuda() for _s in s]

        o, h = self.forward(v, s, c)
        o_size = o.size(1)

        return sequence_size - o_size




