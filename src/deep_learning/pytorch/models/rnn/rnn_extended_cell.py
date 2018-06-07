from src.deep_learning.pytorch.models.rnn.rnn_locked_dropout import RNNLockedDropout
from src.deep_learning.pytorch.models.common.weight_drop import WeightDrop
from src.deep_learning.pytorch.models.rnn.rnn_batch_norm import RNNBatchNorm
from src.deep_learning.pytorch.models.rnn.rnn_layer_norm import RNNLayerNorm
from torch.nn import Module, Conv1d
from torch import cat, transpose


class RNNExtendedCell(Module):
    """
    Basic building block of RNN network.
    Implements:
    - forward dropout,
    - hidden to hidden weight matrix dropout,
    - batch and layer normalization
    - add and concat skip connections
    """
    def __init__(self, cell, in_size, hidden_size, dropout_f, dropout_h, rnn_normalization, skip_mode, use_mc_dropout):
        super().__init__()
        self.skip_mode = skip_mode

        # If dimensions not match we need to pass it through a cnn layer
        if skip_mode == 'add' and in_size != hidden_size:
            self.conv = Conv1d(in_channels=in_size, out_channels=hidden_size, kernel_size=1, stride=1)
        else:
            self.conv = None

        # Pass through the RNN cell
        rnn = cell(input_size=in_size, hidden_size=hidden_size, num_layers=1, dropout=0, batch_first=True,
                   bias=True)

        # Apply dropout on the hidden/hidden weight matrix
        self.rnn = WeightDrop(module=rnn, weights=['weight_hh_l0'], dropout_h=dropout_h, use_mc_dropout=use_mc_dropout)

        # Apply batch normalization
        if rnn_normalization == 'batch_norm':
            self.normalization_layer = RNNBatchNorm(num_features=hidden_size)
        elif rnn_normalization == 'layer_norm':
            self.normalization_layer = RNNLayerNorm(num_features=hidden_size)
        elif rnn_normalization == 'none':
            self.normalization_layer = None
        else:
            raise RuntimeError('Unexpected rnn normalization %s' % rnn_normalization)

        # Apply the same dropout mask in each timestamp for the output
        self.dropout_layer = RNNLockedDropout(dropout=dropout_f, use_mc_dropout=use_mc_dropout)

    def forward(self, x, hidden):
        x_skip = x
        x, hidden = self.rnn(x, hidden)

        if self.normalization_layer is not None:
            x = self.normalization_layer(x)

        x = self.dropout_layer(x)

        if self.skip_mode == 'concat':
            x = cat([x, x_skip], dim=2)
        elif self.skip_mode == 'add':
            if self.conv is not None:
                # Assume input comes as N x L x C
                # Transpose to  N x C x L
                x_skip = transpose(x_skip, 1, 2)
                x_skip = self.conv(x_skip)
                # Transpose back to N x L x C
                x_skip = transpose(x_skip, 1, 2)
            x = x + x_skip

        return x, hidden
