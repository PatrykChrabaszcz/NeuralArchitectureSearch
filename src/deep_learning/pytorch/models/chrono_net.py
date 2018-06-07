import torch
import torch.nn as nn
from src.deep_learning.pytorch.models.model_base import RnnBase
from src.deep_learning.pytorch.models.rnn import RNN
import logging


logger = logging.getLogger(__name__)


class ChronoNet(RnnBase):
    class InceptionBlock(nn.Module):
        def __init__(self, in_size, batch_norm, out_size=32):
            super().__init__()
            self.conv_1 = nn.Conv1d(in_channels=in_size, out_channels=out_size, kernel_size=2, stride=2, padding=0)
            self.conv_2 = nn.Conv1d(in_channels=in_size, out_channels=out_size, kernel_size=4, stride=2, padding=1)
            self.conv_3 = nn.Conv1d(in_channels=in_size, out_channels=out_size, kernel_size=8, stride=2, padding=3)

            self.batch_norm = batch_norm
            if batch_norm:
                self.bnn = nn.BatchNorm1d(num_features=3*out_size)

            self.non_linearity = nn.ReLU

        def forward(self, x):
            # Transpose to  N x C x L
            x = torch.transpose(x, 1, 2)
            x = torch.cat([self.conv_1(x), self.conv_2(x), self.conv_3(x)], dim=1)
            x = self.non_linearity()(x)
            x = torch.transpose(x, 1, 2)
            return x

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inception_block_1 = ChronoNet.InceptionBlock(self.input_size, batch_norm=False, out_size=32)
        self.inception_block_2 = ChronoNet.InceptionBlock(32*3, batch_norm=False, out_size=32)
        self.inception_block_3 = ChronoNet.InceptionBlock(32*3, batch_norm=False, out_size=32)

        cell = RnnBase.cell_mapper[self.rnn_cell_type]
        self.rnn = RNN(cell=cell, in_size=32*3, hidden_size=self.rnn_hidden_size,
                       num_layers=self.rnn_num_layers, dropout_f=self.dropout_f, dropout_h=self.dropout_h,
                       rnn_normalization=self.rnn_normalization, skip_first=self.skip_first, skip_last=self.skip_last,
                       use_mc_dropout=self.use_mc_dropout, skip_mode=self.skip_mode)

        self.fc = nn.Linear(in_features=self.rnn_hidden_size, out_features=self.output_size, bias=True)

    def forward(self, x, hidden, context):
        x = self.inception_block_1(x)
        x = self.inception_block_2(x)
        x = self.inception_block_3(x)

        x, hidden = self.rnn(x, hidden)
        x = self.fc(x)

        return x, hidden

    def offset_size(self, sequence_size):
        assert sequence_size % 8 == 0,  "For this model it is better if sequence size is divisible by 8"
        return sequence_size - sequence_size//8