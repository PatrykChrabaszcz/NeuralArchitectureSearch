from torch.nn import Module, Conv1d, BatchNorm1d, Sequential, ReLU
from torch import transpose


class AllChannelsCNN(Module):
    def __init__(self, in_size, out_size, num_layers, kernel_size, dilation,
                 input_in_rnn_format=False, batch_norm=False):
        super().__init__()

        self.input_in_rnn_format = input_in_rnn_format
        self.sequential = Sequential()

        for i in range(num_layers):
            conv = Conv1d(in_channels=in_size, out_channels=out_size, kernel_size=kernel_size,
                          dilation=dilation, bias=not batch_norm)
            self.sequential.add_module('cnn_c_%s' % i, conv)
            non_lin = ReLU()
            self.sequential.add_module('non_lin_%s' % i, non_lin)
            if batch_norm:
                bnn = BatchNorm1d(out_size)
                self.sequential.add_module('bnn_c_%s' % i, bnn)
            in_size = out_size

    def forward(self, x):
        if self.input_in_rnn_format:
            # RNN format: N x L x C
            # Transpose to CNN format:  N x C x L
            x = transpose(x, 1, 2)

        x = self.sequential(x)

        if self.input_in_rnn_format:
            # RNN format: N x L x C
            # Transpose to CNN format:  N x C x L
            x = transpose(x, 1, 2)

        return x
