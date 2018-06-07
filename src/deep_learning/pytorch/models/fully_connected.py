from src.deep_learning.pytorch.models.model_base import PytorchModelBase
from src.deep_learning.pytorch.models.rnn import RNNLayerNorm
import torch.nn.functional as f
from torch import nn, cat
import logging


logger = logging.getLogger(__name__)


class FCNet(PytorchModelBase):
    @staticmethod
    def add_arguments(parser):
        PytorchModelBase.add_arguments(parser)
        parser.add_argument("fc_num_layers", type=int, default=3,
                            help="Number of fully connected layers")
        parser.add_argument("fc_num_neurons", type=int, default=128,
                            help="Number of neurons in each fc layer")
        parser.add_argument("fc_normalization", type=str, default='none', choices=['none', 'batch_norm', 'layer_norm'],
                            help="Number of neurons in each fc layer")
        return parser

    def __init__(self, fc_num_layers, fc_num_neurons, fc_normalization, **kwargs):
        super().__init__(**kwargs)

        self.fc_layers = []
        self.fc_normalization_layers = []

        input_size = self.input_size if self.use_context == 0 else self.input_size + self.context_size
        for layer_id in range(fc_num_layers):
            in_features = input_size if layer_id == 0 else fc_num_neurons
            out_features = self.output_size if layer_id == fc_num_layers-1 else fc_num_neurons
            fc = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
            self.add_module('fc_%d' % layer_id, fc)
            self.fc_layers.append(fc)

            if layer_id != fc_num_layers - 1:
                if fc_normalization == 'layer_norm':
                    norm_layer = RNNLayerNorm(out_features)
                    self.add_module('layer_norm_%d' % layer_id, norm_layer)
                    self.fc_normalization_layers.append(norm_layer)

    def forward(self, x, hidden, context):
        if self.use_context:
            context = cat([context] * x.size()[1], dim=1)
            x = cat([x, context], dim=2)

        batch_size = x.size(0)
        time_size = x.size(1)
        feature_size = x.size(2)

        # View each timepoints as a separate point
        x = x.view(batch_size*time_size, feature_size)

        for i, fc_layer in enumerate(self.fc_layers):
            x = fc_layer(x)

            # No ReLU in the last layer
            if i != len(self.fc_layers)-1:
                try:
                    x = self.fc_normalization_layers[i](x)
                except:
                    pass
                x = f.relu(x)

        x = x.view(batch_size, time_size, self.output_size)

        return x, hidden

    def offset_size(self, sequence_size):
        return 0