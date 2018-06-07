from torch.nn import Module
from torch.autograd import Variable


# https://github.com/salesforce/awd-lstm-lm/blob/master/locked_dropout.py
class RNNLockedDropout(Module):
    """
    We want to use the same dropout mask for all timepoints. Using this layer we will be able to do so. Dropout masks
    will be different for different examples within the minibatch but will not change in timesteps.
    """
    def __init__(self, dropout, use_mc_dropout):
        super().__init__()
        assert 0.0 <= dropout <= 1.0, 'Dropout has to be in range <0.0, 1.0>'
        self.use_mc_dropout = use_mc_dropout
        self.dropout = dropout

    def forward(self, x):
        if (self.training or self.use_mc_dropout) and self.dropout != 0:
            # Same dropout for all timesteps
            m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1.0 - self.dropout)

            # Handle the case where dropout is 1.0 (Should act as blocking information flow)
            mask = Variable(m, requires_grad=False)
            if self.dropout != 1.0:
                mask /= (1.0 - self.dropout)
            return mask * x
        else:
            return x
