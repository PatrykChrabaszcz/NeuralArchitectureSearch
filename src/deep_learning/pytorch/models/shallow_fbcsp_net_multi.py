from src.deep_learning.pytorch.models.model_base import PytorchModelBase
from src.deep_learning.pytorch.models.cnn import SplitConv
from torch import nn, transpose, from_numpy
from torch.nn import Sequential, BatchNorm1d, ReLU, Dropout
from torch.nn import init, MaxPool1d, AvgPool1d, LogSoftmax
from src.deep_learning.pytorch.models.shallow_fbcsp_net import ShallowFBCSPNet
import logging
import numpy as np
from torch.autograd import Variable
from braindecode.torch_ext.modules import Expression
from braindecode.torch_ext.functions import safe_log, square
import torch

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


class ShallowFBCSPNetMulti(PytorchModelBase):
    """
    From the ConvNet for BrainData Paper
    """

    @staticmethod
    def add_arguments(parser):
        ShallowFBCSPNet.add_arguments(parser)
        parser.add_argument("n_subjects", type=int, default=4,
                            help="TODO")
        return parser

    def __init__(self, n_subjects, **kwargs):
        super().__init__(**kwargs)

        self.n_subjects = n_subjects

        self.models = []
        for i in range(self.n_subjects):
            model = ShallowFBCSPNet(**kwargs)
            self.add_module('model_%d' % i, model)
            self.models.append(model)

    def forward(self, x, hidden, context):
        batch_size = x.size(0)
        time_size = x.size(1)

        inds = []
        outs = []
        for i in range(self.n_subjects):
            ind = (context[:, 0] == i).nonzero().squeeze()
            try:
                x_i = x.index_select(0, ind)
                out, _ = self.models[i](x_i, None, None)
                inds.append(ind)
                outs.append(out)

            except RuntimeError:
                # Because ind is empty
                pass

        # TODO: Replace this concat with something more clever and with less computation
        merged_x = Variable.zeros_like(torch.cat(outs, dim=0))
        for ind, out in zip(inds, outs):
            merged_x[ind] = out

        return merged_x, hidden

    def offset_size(self, sequence_size):
        # # Forward dummy vector and find out what is the output shape
        #
        # v = np.zeros((1, sequence_size, self.input_size), np.float32)
        # v = Variable(from_numpy(v))
        # if next(self.parameters()).is_cuda:
        #     v = v.cuda()
        #
        # o, h = self.forward(v, None, None)
        # o_size = o.size(1)

        return sequence_size - 1# o_size
