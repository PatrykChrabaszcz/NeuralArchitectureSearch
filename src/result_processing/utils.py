import numpy as np


class Metric:
    def __init__(self, dim):
        self.dim = dim
        self._initialize()

    def _initialize(self):
        raise NotImplementedError('Initialize your own metric using this method')

    def append(self, x, y, fresh=False):
        assert x.shape[1] == self.dim
        assert x.shape[0] == y.shape[0]
        if fresh:
            self._initialize()

        self._append(x, y)

    def _append(self, x, y):
        raise NotImplementedError("Append data to your own metric using this method")


class MetricAllEndLast:
    """
    Class that will for a given metric compute metric from all timepoints, from the last timepoint or
    from all chunks last timepoints
    """
    def __init__(self, metric_class, dim):
        self._all = metric_class(dim)
        self._ends = metric_class(dim)
        self._last = metric_class(dim)

    def append(self, x, y, fresh=False):
        self._all.append(x, y, fresh=fresh)
        try:# Handle both regression (2D y) and classification (1D y)
            y = y[[-1], :]
        except IndexError:
            y = y[[-1]]
        self._ends.append(x[[-1], :], y, fresh=fresh)
        self._last.append(x[[-1], :], y, fresh=True)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)

        def f(*args, **kwargs):
            return getattr(self._all, attr)(*args, **kwargs), \
                   getattr(self._ends, attr)(*args, **kwargs), \
                   getattr(self._last, attr)(*args, **kwargs)
        return f


class MetricCount(Metric):
    def _initialize(self):
        self._cnt_all = 0
        self._cnt_end = 0

    def _append(self, x, y):
        self._cnt_all += len(x)
        self._cnt_end += 1

    def cnt(self):
        return self._cnt_all

    def cnt_end(self):
        return self._cnt_end


class MetricSum(Metric):
    """
    Provides functionality to aggregate labels and predictions
    to compute sum or mean of labels or predictions.
    """
    def _initialize(self):
        self._cnt = 0
        self._sum_x = np.zeros([1, self.dim])
        self._sum_y = np.zeros([1, self.dim])

    def _append(self, x, y):
        self._sum_x += np.sum(x, axis=0)
        self._sum_y += np.sum(y, axis=0)
        self._cnt += len(x)

    def sum_x(self):
        return self.sum_x

    def sum_y(self):
        return self.sum_y

    def mean_x(self):
        return self._sum_x / max(1, self._cnt)

    def mean_y(self):
        return self._sum_y / max(1, self._cnt)


class MetricSquareSum(Metric):
    def _initialize(self):
        self._sum_xx = np.zeros([1, self.dim])
        self._sum_yy = np.zeros([1, self.dim])
        self._sum_xy = np.zeros([1, self.dim])
        self._cnt = 0

    def _append(self, x, y):
        assert x.shape[1] == y.shape[1] == self.dim
        assert x.shape[0] == y.shape[0]

        self._sum_xx += np.sum(x**2, axis=0)
        self._sum_yy += np.sum(y**2, axis=0)
        self._sum_xy += np.sum(x*y, axis=0)
        self._cnt += len(x)

    def sum_xx(self):
        return self._sum_xx

    def sum_yy(self):
        return self._sum_yy

    def sum_xy(self):
        return self._sum_xy

    def mean_xx(self):
        return self._sum_xx / max(1, self._cnt)

    def mean_yy(self):
        return self._sum_yy / max(1, self._cnt)

    def mean_xy(self):
        return self._sum_xy / max(1, self._cnt)


class MetricL1Loss(Metric):
    def _initialize(self):
        self._sum_l1 = np.zeros([1, self.dim])
        self._cnt = 0

    def _append(self, x, y):
        assert x.shape[1] == y.shape[1] == self.dim
        assert x.shape[0] == y.shape[0]

        self._sum_l1 += np.sum(np.abs(x - y), axis=0)
        self._cnt += len(x)

    def l1_loss(self):
        return self._sum_l1 / self._cnt


class MetricL2Loss(Metric):
    def _initialize(self):
        self._sum_l2 = np.zeros([1, self.dim])
        self._cnt = 0

    def _append(self, x, y):
        assert x.shape[1] == y.shape[1] == self.dim
        assert x.shape[0] == y.shape[0]

        self._sum_l2 += np.sum((x - y)**2, axis=0)
        self._cnt += len(x)

    def l2_loss(self):
        return self._sum_l2 / self._cnt


class MetricSingleLabelAccuracy(Metric):
    def _initialize(self):
        self._votes = np.zeros([1, self.dim])
        self._power = np.zeros([1, self.dim])
        self._cnt = 0
        self._label = None

    def _append(self, x, y):
        self._votes += np.sum(np.eye(self.dim)[x.argmax(axis=1)], axis=0)
        self._power += np.sum(x, axis=0)
        self._cnt += len(x)
        self._label = y[0]

    def vote_accuracy(self):
        if self._label is None:
            raise RuntimeError('Vote accuracy called when label not know, try to call append(...) first.')
        return 1.0 if np.argmax(self._votes, axis=1) == self._label else 0.0

    def vote_probability(self):
        if self._label is None:
            raise RuntimeError('Vote probability called when label not know, try to call append(...) first.')
        return self._votes / self._cnt

    def log_accuracy(self):
        if self._label is None:
            raise RuntimeError('Log accuracy called when label not know, try to call append(...) first')
        return 1.0 if np.argmax(self._power, axis=1) == self._label else 0.0

    def log_probability(self):
        if self._label is None:
            raise RuntimeError('Log probability called when label not know, try to call append(...) first')
        return self._power / self._cnt


class MetricConfusionMatrix(Metric):
    def _initialize(self):
        pass

    def _append(self, x, y):
        pass


class MetricRunningCovariance(Metric):
    """
    This class is used to compute running correlation statistics between predictions and targets. For each
    (prediction, target) pair we keep the track of required statistics and used them to compute
    corresponding correlation when requested.
    """
    def _initialize(self):
        self.sum = MetricSum(self.dim)
        self.square_sum = MetricSquareSum(self.dim)

    def _append(self, x, y):
        self.sum.append(x, y)
        self.square_sum.append(x, y)

    def corr(self):
        return (self.square_sum.mean_xy() - self.sum.mean_x() * self.sum.mean_y()) / \
               np.sqrt(np.clip(self.var_x() * self.var_y(), 1e-5, None))

    def var_x(self):
        return self.square_sum.mean_xx() - self.sum.mean_x()**2

    def var_y(self):
        return self.square_sum.mean_yy() - self.sum.mean_y()**2


# Would be more professional to write unit tests.
# Right now only some simple user assisted sanity checks are provided.
if __name__ == '__main__':
    def test_running_covariance():
        np.random.seed(0)
        x = np.random.uniform(-100, 1000, size=[100000, 5])
        y = np.random.uniform(-100, 1000, size=[100000, 5])

        c = []
        for i in range(5):
            c.append(np.cov(x[:, i], y[:, i]))

        rc = MetricRunningCovariance(dim=5)
        rc_online = MetricRunningCovariance(dim=5)
        rc.append(x, y)
        rc_online.append(x[:x.shape[0] // 2, :], y[:y.shape[0] // 2, :])
        rc_online.append(x[x.shape[0] // 2:, :], y[y.shape[0] // 2:, :])

        x_var = [e[0][0] for e in c]

        print('\nX_Var')
        print('Numpy:')
        print(x_var)
        print('Running Convariance Class:')
        print(rc.var_x())
        print('Running Convariance Class Online:')
        print(rc_online.var_x())

        xy_covar = [e[0][1] / np.sqrt(e[0][0] * e[1][1]) for e in c]

        print('\nXY_Corr')
        print('Numpy:')
        print(xy_covar)
        print('Running Convariance Class:')
        print(rc.corr())
        print('Running Convariance Class Online:')
        print(rc_online.corr())

    def test_regression_metrics():
        from src.result_processing import MultiLabelRegressionMetrics
        metrics = MultiLabelRegressionMetrics('Metrics', 2, 0)

        example_ids = [1, 2, 3]
        output = np.array([
            [[1, 1], [1, 1], [1, 1], [1, 2]],
            [[2, 2], [2, 2], [2, 2], [2, 2]],
            [[3, 2], [3, 3], [3, 3], [3, 2]]
        ])
        labels = np.array([
            [[1, 1], [1, 1], [1, 1], [1, 1]],
            [[2, 2], [2, 2], [2, 2], [2, 2]],
            [[3, 3], [3, 3], [3, 3], [3, 3]]
        ])
        metrics.append_results(example_ids, output, labels, 0)

        results = metrics.get_summarized_results()
        print(results)

    def test_metric_all_end_last():
        np.random.seed(0)
        dim = 5
        x = np.random.uniform(-100, 1000, size=[100000, dim])
        y = np.random.uniform(-100, 1000, size=[100000, dim])

        m = MetricAllEndLast(metric_class=MetricSum, dim=dim)
        m_all = MetricSum(dim=dim)
        m_end = MetricSum(dim=dim)
        m_last = MetricSum(dim=dim)

        r = 10
        s = x.shape[0]
        for i in range(r):
            s_s = int(s/r * i)
            s_e = int(s/r * (i+1))

            x_s = x[s_s:s_e, :]
            y_s = y[s_s:s_e, :]

            m.append(x_s, y_s)
            m_all.append(x_s, y_s)
            m_end.append(x_s[[-1], :], y_s[[-1], :])
            m_last.append(x_s[[-1], :], y_s[[-1], :], fresh=True)

        print(m.mean_x())
        print(m_all.mean_x())
        print(m_end.mean_x())
        print(m_last.mean_x())

    test_metric_all_end_last()
    #test_running_covariance()
    #test_regression_metrics()


