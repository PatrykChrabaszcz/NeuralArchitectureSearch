from torch.nn import Module, BatchNorm1d


class RNNBatchNorm(Module):
    """
    Module implemented for convenience. It uses underlying BatchNorm1d
    implementation. RNN input is shaped as [batch_size x time x features].
    Default (designed for cnn) BN implementation needs it to have the shape:
    [batch_size x features x time] or [batch_size*time x features].
    This layer manages this reshaping: reshape_to_cnn -> BN -> reshape_to_rnn
    Normalization statistics for each feature are computed across all examples
    and all time-points. Some papers show the advantage of keeping separate
    statistics for each time-point, however this layer does not implement that.
    Side Note:
        In most (if not all) of our networks if we implement batch_norm we only
        normalize outputs from RNN layers (after non-linearity). We do not normalize
        internal (within the cell) activations, with current cuDNN implementation
        it is not possible. We could do that in custom cells but then we lose
        cuDNN speed up.
        https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/
    """
    def __init__(self, num_features):
        super().__init__()

        self.bn = BatchNorm1d(num_features)
        self.bn.weight.data.fill_(1)

    def forward(self, x):
        x.contiguous()
        n, t = x.size(0), x.size(1)
        x = x.view(n * t, -1)
        x = self.bn(x)
        x = x.view(n, t, -1)
        return x

# TO DO: CHECK IT
# if __name__ == '__main__':
#     import numpy as np
#     batch = np.array([
#         [[1, 1, 1], [2, 2, 2], [3, 3, 4]],
#         [[1, 1, 1], [2, 2, 2], [3, 3, 3]]]).astype(np.float32)
#
#     batch = Variable(torch.from_numpy(batch))
#
#     d = RNNBatchNorm(3)
#
#     l = RNNLayerNorm(3)
#     print([p for p in d.parameters()])
#     print('Input batch')
#     print(batch)
#
#     print('Batch Norm output')
#     print(d(batch))
#
#     print('Layer Norm output')
#     print(l(batch))
