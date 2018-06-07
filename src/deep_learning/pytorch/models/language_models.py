import torch.nn as nn
from src.dl_pytorch.model import RnnBase
from src.deep_learning.pytorch.models.utils import RNN


class WikiTextRNN(RnnBase):
    @staticmethod
    def add_arguments(parser):
        RnnBase.add_arguments(parser)
        parser.add_argument("tie_weights", type=int, default=1, choices=[0, 1],
                            help="TODO")
        parser.add_argument("embedding_size", type=int, default=400,
                            help="Number of layers in the RNN network.")
        return parser

    def __init__(self, tie_weights, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.drop = nn.Dropout(self.dropout_f)
        self.encoder = nn.Embedding(self.input_size, embedding_size)

        cell = RnnBase.cell_mapper[self.rnn_cell_type]
        self.rnn = RNN(cell=cell, in_size=embedding_size, hidden_size=self.rnn_hidden_size,
                       num_layers=self.rnn_num_layers,
                       dropout_f=self.dropout_f, dropout_h=self.dropout_h,
                       batch_norm=self.batch_norm, skip_mode=self.skip_mode)

        self.decoder = nn.Linear(self.rnn_hidden_size, self.input_size)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462

        if tie_weights:
            if self.rnn_hidden_size != embedding_size:
                raise ValueError('When using the tied flag, hidden size must be equal to embedding size')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, context):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def offset_size(self, sequence_size):
        return 0