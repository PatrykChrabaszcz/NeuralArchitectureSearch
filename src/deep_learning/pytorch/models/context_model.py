from src.deep_learning.pytorch.models.model_base import PytorchModelBase
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import logging
import torch
import numpy as np


logger = logging.getLogger(__name__)


class ContextModel(PytorchModelBase):
    @staticmethod
    def add_arguments(parser):
        PytorchModelBase.add_arguments(parser)
        parser.add_argument("num_layers", type=int, default=1, help="Number of hidden layers (0 or higher)")
        parser.add_argument("hidden_size", type=int, default=32, help="Number of features in each hidden layer")
        return parser

    def __init__(self, num_layers, hidden_size, **kwargs):
        super().__init__(**kwargs)

        assert self.context_size > 0, 'Context model needs context'

        self.layers = []
        in_size = self.context_size
        for i in range(num_layers):
            fc = nn.Linear(in_features=in_size, out_features=hidden_size, bias=True)
            self.add_module('fc_%d' % i, fc)
            self.layers.append(fc)
            in_size = hidden_size

        self.layer_out = nn.Linear(in_features=in_size, out_features=self.output_size, bias=True)

    def forward(self, x, hidden, context):
        n, t = x.size(0), x.size(1)

        # Use only context
        x = context

        x = torch.cat([x] * t, dim=1)

        f = x.size(2)
        x = x.view(n*t, f)
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)

        x = self.layer_out(x)
        x = x.view(n, t, f)

        return x, hidden

    def initial_state(self):
        return None

    # Dummy
    def export_state(self, states):
        l = states.cpu().data.numpy()[0]
        return [None] * l

    # Dummy
    def import_state(self, states):
        return Variable(torch.from_numpy(np.array([len(states)])), requires_grad=False)

    def offset_size(self, sequence_size):
        return 0
