from src.deep_learning.pytorch.models.model_base import PytorchModelBase
from src.deep_learning.pytorch.models.cnn import SplitConv
from torch import nn, transpose, from_numpy, sort
from torch import max as torch_max
from torch.nn import Sequential, BatchNorm1d, ReLU, Dropout
from torch.nn import init, MaxPool1d, AvgPool1d, LogSoftmax
import logging
import numpy as np
from torch.autograd import Variable
from braindecode.torch_ext.modules import Expression
from braindecode.torch_ext.functions import safe_log, square


nonlin_dict = {
    'square': square,
    'safe_log': safe_log
}
pool_dict = {
    "max": MaxPool1d,
    "mean": AvgPool1d
}


logger = logging.getLogger(__name__)


# remove empty dim at end and potentially remove empty time dim
# do not just use squeeze as we never want to remove first dim
def _squeeze_final_output(x):
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x


def _transpose_time_to_spat(x):
    return x.permute(0, 3, 2, 1)


class ShallowFBCSPNet(PytorchModelBase):
    """
    From the ConvNet for BrainData Paper
    """

    @staticmethod
    def add_arguments(parser):
        PytorchModelBase.add_arguments(parser)
        parser.add_argument("n_filters_time", type=int, default=40,
                            help="TODO")
        parser.add_argument("filter_time_length", type=int, default=25,
                            help="TODO")
        parser.add_argument("n_filters_spat", type=int, default=40,
                            help="TODO")
        parser.add_argument("pool_time_length", type=int, default=75,
                            help="TODO")
        parser.add_argument("pool_time_stride", type=int, default=15,
                            help="TODO")
        parser.add_argument("conv_nonlin", type=str, default="square", choices=nonlin_dict.keys(),
                            help="TODO")
        parser.add_argument("pool_mode", type=str, default="mean",
                            help="TODO")
        parser.add_argument("pool_nonlin", type=str, default="safe_log",
                            help="TODO")
        parser.add_argument("batch_norm", type=int, default=1,
                            help="TODO")
        parser.add_argument("batch_norm_alpha", type=float, default=0.1,
                            help="TODO")
        parser.add_argument("drop_prob", type=float, default=0.5,
                            help="TODO")
        parser.add_argument("final_conv_length", type=int, default=30,
                            help="TODO")

        return parser

    def __init__(self, n_filters_time, filter_time_length, n_filters_spat,
                 pool_time_length, pool_time_stride, conv_nonlin, pool_mode, pool_nonlin,
                 batch_norm, batch_norm_alpha, drop_prob, final_conv_length, **kwargs):
        super().__init__(**kwargs)

        self.sequential = Sequential()
        split_cnn = SplitConv(in_size=self.input_size, middle_size=n_filters_time, out_size=n_filters_spat,
                              time_kernel_size=filter_time_length, input_in_rnn_format=False)
        self.sequential.add_module('split_cnn', split_cnn)

        if batch_norm:
            bn = BatchNorm1d(n_filters_spat)
            self.sequential.add_module('batch_norm', bn)

        non_lin = Expression(square)
        self.sequential.add_module('non_lin_0', non_lin)

        pool = AvgPool1d(kernel_size=pool_time_length, stride=pool_time_stride)
        self.sequential.add_module('pool_1', pool)

        non_lin = Expression(safe_log)
        self.sequential.add_module('non_lin_1', non_lin)

        dropout = Dropout(p=drop_prob)
        self.sequential.add_module('dropout', dropout)

        conv = nn.Conv1d(in_channels=n_filters_spat, out_channels=self.output_size, kernel_size=final_conv_length,
                         bias=True)

        self.sequential.add_module('conv', conv)

    def forward(self, x, hidden, context):
        # Input is given as N x L x C
        # ConvNets expect N x C x L
        x = transpose(x, 1, 2)
        x = self.sequential(x)
        # Convert back to N x L x C
        x = transpose(x, 1, 2)
        # x = x[:, i:10, :]
        x = x.contiguous()

        return x, hidden

    def offset_size(self, sequence_size):
        # Forward dummy vector and find out what is the output shape

        v = np.zeros((1, sequence_size, self.input_size), np.float32)
        v = Variable(from_numpy(v))
        if next(self.parameters()).is_cuda:
            v = v.cuda()

        o, h = self.forward(v, None, None)
        o_size = o.size(1)

        return sequence_size - o_size
