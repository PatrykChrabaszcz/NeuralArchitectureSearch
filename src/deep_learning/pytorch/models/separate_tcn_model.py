import torch.nn as nn
from torch.nn.utils import weight_norm
from src.deep_learning.pytorch.models.model_base import PytorchModelBase
from torch import transpose
import time
# Copy from https://github.com/locuslab/TCN/blob/master/TCN/tcn.py

class Chomp2d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, (1, kernel_size),
                                           stride=(1, stride), padding=(0, padding), dilation=(1, dilation)))
        self.chomp1 = Chomp2d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(dropout)

        self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs, (1, kernel_size),
                                           stride=(1, stride), padding=(0, padding), dilation=(1, dilation)))

        self.chomp2 = Chomp2d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv2d(n_inputs, n_outputs, kernel_size=(1, 1)) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class SeparateTemporalConvNet(PytorchModelBase):
    @staticmethod
    def add_arguments(parser):
        PytorchModelBase.add_arguments(parser)
        parser.add_argument("embedding_size", type=int, default=16,
                            help="TODO")
        parser.add_argument("dropout", type=float, default=0.2,
                            help="TODO")
        parser.add_argument("kernel_size", type=int, default=2,
                            help="TODO")
        parser.add_argument("num_channels", type=int, default=128,
                            help="TODO")
        parser.add_argument("num_levels", type=int, default=3,
                            help="TODO")
        return parser

    def __init__(self, embedding_size, num_channels, num_levels, kernel_size=2, dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        layers = []
        num_channels = [num_channels] * num_levels

        self.embedding_layer = nn.Conv1d(in_channels=self.input_size, out_channels=embedding_size, kernel_size=1)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = 1 if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

        self.fc = nn.Linear(in_features=num_channels[-1]*embedding_size, out_features=self.output_size, bias=True)

        self.min_len = 1
        for i in range(num_levels):
            dilation = 2 ** i
            self.min_len += 2 * (kernel_size - 1) * dilation

    def forward(self, x, hidden, context):
        batch_size = x.size(0)
        time_size = x.size(1)

        # RNN format: N x L x C
        # Transpose to CNN format:  N x C x L
        x = transpose(x, 1, 2)

        x = self.embedding_layer(x)

        # We need the same cnn for each channel so lets convert it to 2d (Conv2d expects N x C x H x W)
        # We need to convert into: N x 1 x in_size x L
        x = x.unsqueeze(1)

        x = self.network(x)
        # Now it should be N x out_size x in_size x L

        out_size = 1 + max(0, time_size-self.min_len)

        x = x[:, :, :, -out_size:].contiguous()

        # Convert to N x out_size*in_size x L
        x = x.view(x.size(0), x.size(1)*x.size(2), x.size(3))

        # RNN format: N x L x C
        x = transpose(x, 1, 2).contiguous()

        fc_out = self.fc(x.view(batch_size * out_size, x.size(2)))

        fc_out = fc_out.view(batch_size, out_size, fc_out.size(1)).contiguous()

        return fc_out,  hidden

    def offset_size(self, sequence_size):
        out_size = 1 + max(0, sequence_size-self.min_len)
        return sequence_size-out_size
