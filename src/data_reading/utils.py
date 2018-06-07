import numpy as np


class RunningStatistics:
    """
    Used to compute per channel and aggregated mean and variance from multiple data .
    """
    def __init__(self, dim, time_dimension_first=True):
        self.dim = 0
        self.cnt = np.float32(0)

        if time_dimension_first:
            self.axis = 0
            self.sum = np.zeros([1, dim], dtype=np.float32)
            self.square_sum = np.zeros([1, dim], dtype=np.float32)
        else:
            self.axis = 1
            self.sum = np.zeros([dim, 1], dtype=np.float32)
            self.square_sum = np.zeros([dim, 1], dtype=np.float32)

    def append(self, x):
        assert len(x.shape) == 2, 'Can only process 2D arrays'
        self.sum += np.sum(x, axis=self.axis, keepdims=True)
        self.square_sum += np.sum(x**2, axis=self.axis, keepdims=True)
        self.cnt += x.shape[self.axis]

    def var_vector(self):
        return self.square_sum/self.cnt - (self.sum/self.cnt) ** 2

    def var_scalar(self):
        return np.mean(self.square_sum) / self.cnt - (np.mean(self.sum) / self.cnt) ** 2

    def mean_vector(self):
        return self.sum / self.cnt

    def mean_scalar(self):
        return np.mean(self.sum) / self.cnt