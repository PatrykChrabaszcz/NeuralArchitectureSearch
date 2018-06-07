from src.deep_learning.pytorch.models.rnn import RNNLockedDropout, RNN, RNNLayerNorm
from src.deep_learning.pytorch.models.model_base import RnnBase
import torch.nn.functional as f
import torch.nn as nn
from torch.autograd import Variable
import logging
import torch


logger = logging.getLogger(__name__)


class SplitRNN(RnnBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_dropout_layer = LockedDropout(self.dropout_i, use_mc_dropout=self.use_mc_dropout)

        cell = RnnBase.cell_mapper[self.rnn_cell_type]

        # Main RNN module
        assert self.use_context, 'This model only works with the context'

        self.rnn_m = RNN(cell=cell, in_size=self.input_size, hidden_size=self.rnn_hidden_size,
                         num_layers=self.rnn_num_layers, dropout_f=self.dropout_f, dropout_h=self.dropout_h,
                         rnn_normalization=self.rnn_normalization, dilation=self.rnn_dilation, skip_mode=self.skip_mode,
                         skip_first=self.skip_first, skip_last=self.skip_last, use_mc_dropout=self.use_mc_dropout)
        self.rnn_f = RNN(cell=cell, in_size=self.input_size, hidden_size=self.rnn_hidden_size,
                         num_layers=self.rnn_num_layers, dropout_f=self.dropout_f, dropout_h=self.dropout_h,
                         rnn_normalization=self.rnn_normalization, dilation=self.rnn_dilation, skip_mode=self.skip_mode,
                         skip_first=self.skip_first, skip_last=self.skip_last, use_mc_dropout=self.use_mc_dropout)

        self.fc_m = nn.Linear(in_features=self.rnn_hidden_size, out_features=self.output_size, bias=True)
        self.fc_f = nn.Linear(in_features=self.rnn_hidden_size, out_features=self.output_size, bias=True)

    def forward(self, x, hidden, context):
        batch_size = x.size(0)
        time_size = x.size(1)

        # Dropout on the input features
        x = self.input_dropout_layer(x)

        # Divide male and female data
        ind_m = (context[:, 0, 0] == 1.0).nonzero().squeeze()
        ind_f = (context[:, 0, 0] != 1.0).nonzero().squeeze()

        x_m = x.index_select(0, ind_m)
        x_f = x.index_select(0, ind_f)

        hidden_m = [h.index_select(1, ind_m) for h in hidden]
        hidden_f = [h.index_select(1, ind_f) for h in hidden]
        #
        # out = Variable.zeros_like(x)
        # out[ind_m] = x_m
        # out[ind_f] = x_f

        # Rnn with all the features
        lstm_out_m, hidden_m = self.rnn_m(x_m, hidden_m)
        lstm_out_f, hidden_f = self.rnn_f(x_f, hidden_f)

        fc_out_m = self.fc_m(lstm_out_m.view(lstm_out_m.size(0) * time_size, lstm_out_m.size(2)))
        fc_out_f = self.fc_f(lstm_out_f.view(lstm_out_f.size(0) * time_size, lstm_out_f.size(2)))
        fc_out_m = fc_out_m.view(lstm_out_m.size(0), time_size, fc_out_m.size(1))
        fc_out_f = fc_out_f.view(lstm_out_f.size(0), time_size, fc_out_f.size(1))

        # TODO: Replace this concat with something more clever and with less computation
        merged_x = Variable.zeros_like(torch.cat([fc_out_m, fc_out_f], dim=0))
        merged_x[ind_m] = fc_out_m
        merged_x[ind_f] = fc_out_f

        for h, hm, hf in zip(hidden, hidden_m, hidden_f):
            h.data[0][ind_m.data] = hm.data[0]
            h.data[0][ind_f.data] = hf.data[0]

        # fc_out = self.fc(merged_x.view(batch_size * time_size, merged_x.size(2)))

        return merged_x, hidden

    def offset_size(self, sequence_size):
        return 0
