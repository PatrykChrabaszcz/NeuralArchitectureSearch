from torch.nn import Module, Conv2d
from torch import transpose
import torch.nn.functional as f

class SplitConv(Module):
    """
    ...
    """
    def __init__(self, in_size, middle_size, out_size, time_kernel_size, out_bias=False, input_in_rnn_format=False):
        super().__init__()
        self.input_in_rnn_format = input_in_rnn_format

        self.conv_time = Conv2d(in_channels=1, out_channels=middle_size, kernel_size=(1, time_kernel_size))
        self.conv_spatial = Conv2d(in_channels=middle_size, out_channels=out_size, kernel_size=(in_size, 1),
                                   bias=out_bias)

    def forward(self, x):
        if self.input_in_rnn_format:
            # RNN format: N x L x C
            # Transpose to CNN format:  N x C x L
            x = transpose(x, 1, 2)

        # We need the same cnn for each channel so lets convert it to 2d (Conv2d expects N x C x H x W)
        # We need to convert into: N x 1 x C x L
        x = x.unsqueeze(1)

        x = self.conv_time(x)
        x = self.conv_spatial(x)

        # Now it should be N x C x 1 x L
        # Convert to N x C x L
        x = x.squeeze(2)

        if self.input_in_rnn_format:
            # Transpose to RNN format:  N x L x C
            x = transpose(x, 1, 2)

        return x
