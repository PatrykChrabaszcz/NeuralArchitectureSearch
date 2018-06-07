from torch.nn import Module, BatchNorm1d, Conv2d, ReLU, Sequential, BatchNorm2d
from torch import transpose


class SeparateChannelCNN(Module):
    """
    ...
    """
    def __init__(self, out_size, num_layers, kernel_size, dilation, input_in_rnn_format,
                 batch_norm=False):
        super().__init__()

        self.input_in_rnn_format = input_in_rnn_format

        self.sequential = Sequential()
        in_channels = 1
        for i in range(num_layers):
            conv = Conv2d(in_channels=in_channels, out_channels=out_size, kernel_size=(1, kernel_size),
                          dilation=(1, dilation), bias=not batch_norm)
            self.sequential.add_module('conv_%d' % i, conv)
            non_lin = ReLU()
            self.sequential.add_module('relu_%d' % i, non_lin)
            if batch_norm:
                bn = BatchNorm2d(out_size)
                self.sequential.add_module('bn_%d' % i, bn)
            in_channels = out_size

    def forward(self, x):
        if self.input_in_rnn_format:
            # RNN format: N x L x in_size
            # Transpose to CNN format:  N x in_size x L
            x = transpose(x, 1, 2)

        # We need the same cnn for each channel so lets convert it to 2d (Conv2d expects N x C x H x W)
        # We need to convert into: N x 1 x in_size x L
        x = x.unsqueeze(1)

        x = self.sequential(x)

        # Now it should be N x out_size x in_size x L
        # Convert to N x out_size*in_size x L
        x = x.view(x.size(0), x.size(1)*x.size(2), x.size(3))

        if self.input_in_rnn_format:
            # Transpose to RNN format:  N x L x out_size*in_size
            x = transpose(x, 1, 2)
        return x
