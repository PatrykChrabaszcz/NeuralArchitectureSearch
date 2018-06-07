import numpy as np
from src.utils import save_dict


class RunningNormStats:
    def __init__(self, ch_names):
        self.ch_names = ch_names
        self.cnt = np.float32(0)
        self.sum = np.zeros([len(ch_names), 1], dtype=np.float32)
        self.sum_square = np.zeros([len(ch_names), 1], dtype=np.float32)

    def append_data(self, data):
        self.cnt += data.shape[1]
        self.sum += np.sum(data, axis=1, keepdims=True)
        self.sum_square += np.sum(np.square(data), axis=1, keepdims=True)

    @property
    def mean(self):
        return self.sum / self.cnt

    @property
    def stdv(self):
        return np.sqrt(np.maximum((self.sum_square / self.cnt - np.square(self.mean)), 1e-6))

    def save(self, path):
        info = dict()
        m = self.mean
        s = self.stdv
        for i, ch_name in enumerate(self.ch_names):
            info[ch_name] = {
                'mean_microvolt': float(m[i][0]),
                'stdv_microvolt': float(s[i][0])
            }
        save_dict(info, path)
