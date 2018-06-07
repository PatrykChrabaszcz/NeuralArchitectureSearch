from src.result_processing.metrics_base import MetricsBase, MetricsExampleBase
from src.result_processing.utils import MetricRunningCovariance, MetricL1Loss, MetricL2Loss, MetricSum, MetricCount, \
    MetricAllEndLast
import numpy as np


class Example(MetricsExampleBase):
    def __init__(self, example_id, output_size, skip_first_cnt):
        super().__init__(example_id=example_id, output_size=output_size, skip_first_cnt=skip_first_cnt)

        self._cnt = MetricCount(output_size)
        self._l2_loss = MetricL2Loss( dim=output_size)
        self._running_covariance = MetricRunningCovariance(dim=output_size)
        self._variables = MetricSum(dim=output_size)

        self.metrics = [self._cnt,  self._l2_loss, self._running_covariance, self._variables]

    # Output should have shape [timepoints x labels_cnt]
    def _append(self, output, labels):
        assert output.shape == labels.shape, 'Shape of outputs and labels does not match'
        for metric in self.metrics:
            metric.append(output, labels)

    def stats(self):
        l2_loss_all = self._l2_loss.l2_loss()
        prediction_all = self._variables.mean_x()
        target_all = self._variables.mean_y()
        correlation_all = self._running_covariance.corr()
        prediction_var_all = self._running_covariance.var_x()
        target_var_all = self._running_covariance.var_y()

        # Separate statistics for each label
        res = {
            'l2_loss_all': l2_loss_all.flatten().tolist(),
            'prediction_all': prediction_all.flatten().tolist(),
            'target_all': target_all.flatten().tolist(),
            'correlation_all': correlation_all.flatten().tolist(),

            'prediction_variance_all': prediction_var_all.flatten().tolist(),
            'target_variance_all': target_var_all.flatten().tolist(),
        }
        # Aggregate statistics
        for key, item in list(res.items()):
            res['%s_mean' % key] = sum(item)/len(item)

        res['cnt_all'] = self._cnt.cnt()

        return res

    @staticmethod
    def average_stats(stats):
        res = {}
        example_cnt = len(stats)

        if example_cnt == 0:
            return res

        cnt_all = sum(s['cnt_all'] for s in stats)

        res['example_cnt'] = int(example_cnt)
        res['cnt_all'] = int(cnt_all)

        # Simply compute the mean for all fields
        # It is not weighted mean, each example has the same vote power
        for key in stats[0].keys():
            if 'cnt' in key:
                continue
            try:
                res[key] = sum([s[key] for s in stats]) / example_cnt
            except TypeError:
                res[key] = [sum(b)/example_cnt for b in zip(*[s[key] for s in stats])]

        # In addition compute covariance between the samples
        dim = len(res['prediction_all'])
        cov = MetricRunningCovariance(dim=dim)

        x = np.concatenate([np.array(s['prediction_all']) for s in stats]).reshape([example_cnt, dim])
        y = np.concatenate([np.array(s['target_all']) for s in stats]).reshape([example_cnt, dim])
        cov.append(x, y)

        res['B_correlation_all'] = cov.corr().flatten().tolist()
        res['B_prediction_variance_all'] = cov.var_x().flatten().tolist()
        res['B_target_variance_all'] = cov.var_y().flatten().tolist()
        res['B_l2_loss_all'] = np.mean((x-y)**2)

        return res


class MetricsMinimalRegression(MetricsBase):
    def __init__(self, name, output_size, skip_first_cnt):
        super().__init__(Example, name, output_size, skip_first_cnt)
