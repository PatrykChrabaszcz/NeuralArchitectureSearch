from torch.nn import Module, Parameter
from torch import ones, zeros


class RNNLayerNorm(Module):
    """
    Class implementing normalization over the last dimension. In case of RNN,
    when input is shaped as: [batch_size x time x features] this layer will compute
    normalization statistics over the features for each time-point and each example
    separately. Does not need to store running mean and running std for a later
    inference since normalization of one sample is not directly affected by other
    samples in the mini-batch.
    """
    def __init__(self, num_features):
        super().__init__()
        self.gamma = Parameter(ones(num_features))
        self.beta = Parameter(zeros(num_features))
        self.eps = 1e-5

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.gamma * (x - mean) / (std + self.eps) + self.beta
