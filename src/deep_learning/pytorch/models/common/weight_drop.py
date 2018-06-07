from torch.nn import Module, Parameter, RNNBase
import torch.nn.functional as f


# https://github.com/salesforce/awd-lstm-lm/blob/master/weight_drop.py
class WeightDrop(Module):
    """
    CuDNN implementation of RNN networks is much faster but also limited. We are not able to specify the dropout
    on hidden to hidden connections. If we use an implementation that allow to do that we will lose a lot on speed.
    As a solution we use DropConnect on Hidden to Hidden matrices. This will apply the same dropout mask for
    every timepoint and every example within the minibatch.
    """
    def __init__(self, module, weights, dropout_h=0, use_mc_dropout=False):
        super().__init__()
        self.module = module
        self.weights = weights
        self.dropout_h = dropout_h
        self.use_mc_dropout = use_mc_dropout

        # Only drop the weights before calling the module if dropout_h is set to non zero value
        if self.dropout_h != 0:
            self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            if self.use_mc_dropout:
                w = f.dropout(raw_w, p=self.dropout_h, training=True)
            else:
                w = f.dropout(raw_w, p=self.dropout_h, training=self.training)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        if self.dropout_h != 0.:
            self._setweights()
        return self.module.forward(*args)
