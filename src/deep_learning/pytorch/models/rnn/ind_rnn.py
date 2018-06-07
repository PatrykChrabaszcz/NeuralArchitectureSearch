from torch.nn import Module, RNN, Parameter
from torch import diag,  FloatTensor


# Independent RNN as in the recent paper
class IndRNN(Module):
    def __init__(self, hidden_size, *args, **kwargs):
        super().__init__()
        self.module = RNN(hidden_size=hidden_size, *args, **kwargs, nonlinearity='relu')

        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        # I'm not sure what is going on here, this is what weight_drop does so I stick to it
        self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        # We need to register it in this module to make it work with weight dropout
        w_hh = FloatTensor(hidden_size).type_as(getattr(self.module, 'weight_hh_l0').data)
        w_hh.uniform_(-1, 1)

        getattr(self.module, 'bias_ih_l0').data.fill_(0)
        getattr(self.module, 'bias_hh_l0').data.fill_(0)

        self.register_parameter(name='weight_hh_l0', param=Parameter(w_hh))
        del self.module._parameters['weight_hh_l0']

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return

    def _setweights(self):
        w_hh = getattr(self, 'weight_hh_l0')
        w_hh = diag(w_hh)
        setattr(self.module, 'weight_hh_l0', w_hh)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)