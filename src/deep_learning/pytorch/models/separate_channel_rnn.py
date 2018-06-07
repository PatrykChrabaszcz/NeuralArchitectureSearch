from src.deep_learning.pytorch.models.rnn import RNNLockedDropout, RNN
from src.deep_learning.pytorch.models.model_base import RnnBase
from src.deep_learning.pytorch.models.cnn import SplitConv
import torch.nn.functional as f
import torch.nn as nn
import logging
import torch


logger = logging.getLogger(__name__)


class SeparateChannelRNN(RnnBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_dropout_layer = RNNLockedDropout(self.dropout_i, use_mc_dropout=self.use_mc_dropout)

        cell = RnnBase.cell_mapper[self.rnn_cell_type]

        # Add channels with context if requested

        ch = 40
        self.separate_cnn = SplitConv(self.input_size, 40, ch, time_kernel_size=25, out_bias=True,
                                      input_in_rnn_format=True)
        input_size = ch if not self.use_context else ch + self.context_size
        # Main RNN module

        self.rnn = RNN(cell=cell, in_size=input_size, hidden_size=self.rnn_hidden_size,
                       num_layers=self.rnn_num_layers, dropout_f=self.dropout_f, dropout_h=self.dropout_h,
                       rnn_normalization=self.rnn_normalization, dilation=self.rnn_dilation, skip_mode=self.skip_mode,
                       skip_first=self.skip_first, skip_last=self.skip_last, use_mc_dropout=self.use_mc_dropout)

        self.fc = nn.Linear(in_features=self.rnn_hidden_size, out_features=self.output_size, bias=True)

    def forward(self, x, hidden, context):
        # Dropout on the input features
        x = self.input_dropout_layer(x)
        x = self.separate_cnn(x)
        x = f.relu(x)

        batch_size = x.size(0)
        time_size = x.size(1)
        if self.use_context:
            # Repeat context for each time-step
            context = torch.cat([context] * time_size, dim=1)
            x = torch.cat([x, context], dim=2)
            assert list(x.size()) == [batch_size, time_size, self.input_size + self.context_size]


        # Rnn with extended cell
        lstm_out, hidden = self.rnn(x, hidden)
        lstm_out = lstm_out.contiguous()

        fc_out = self.fc(lstm_out.view(batch_size*time_size, lstm_out.size(2)))
        fc_out = fc_out.view(batch_size, time_size, fc_out.size(1))

        return fc_out, hidden

    def offset_size(self, sequence_size):
        # Since we do not use conv nets output size is equal to the input size
        return 24
