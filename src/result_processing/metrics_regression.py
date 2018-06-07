from src.result_processing.metrics_base import MetricsBase, MetricsExampleBase
from src.result_processing.utils import MetricRunningCovariance, MetricL1Loss, MetricL2Loss, MetricSum, MetricCount, \
    MetricAllEndLast
import numpy as np


class Example(MetricsExampleBase):
    def __init__(self, example_id, output_size, skip_first_cnt):
        super().__init__(example_id=example_id, output_size=output_size, skip_first_cnt=skip_first_cnt)

        self._cnt = MetricCount(output_size)
        self._l1_loss = MetricAllEndLast(metric_class=MetricL1Loss, dim=output_size)
        self._l2_loss = MetricAllEndLast(metric_class=MetricL2Loss, dim=output_size)
        self._running_covariance = MetricAllEndLast(metric_class=MetricRunningCovariance, dim=output_size)
        self._variables = MetricAllEndLast(metric_class=MetricSum, dim=output_size)

        self.metrics = [self._cnt, self._l1_loss, self._l2_loss, self._running_covariance, self._variables]

    # Output should have shape [timepoints x labels_cnt]
    def _append(self, output, labels):
        assert output.shape == labels.shape, 'Shape of outputs and labels does not match'
        for metric in self.metrics:
            metric.append(output, labels)

    def stats(self):
        l1_loss_all, l1_loss_end, l1_loss_last = self._l1_loss.l1_loss()
        l2_loss_all, l2_loss_end, l2_loss_last = self._l2_loss.l2_loss()
        prediction_all, prediction_end, prediction_last = self._variables.mean_x()
        target_all, target_end, target_last = self._variables.mean_y()
        correlation_all, correlation_end, _ = self._running_covariance.corr()
        prediction_var_all, prediction_var_end, _ = self._running_covariance.var_x()
        target_var_all, target_var_end, _ = self._running_covariance.var_y()

        # Separate statistics for each label
        res = {

            'l1_loss_all': l1_loss_all.flatten().tolist(),
            'l1_loss_end': l1_loss_end.flatten().tolist(),
            'l1_loss_last': l1_loss_last.flatten().tolist(),

            'l2_loss_all': l2_loss_all.flatten().tolist(),
            'l2_loss_end': l2_loss_end.flatten().tolist(),
            'l2_loss_last': l2_loss_last.flatten().tolist(),

            'prediction_all': prediction_all.flatten().tolist(),
            'prediction_end': prediction_end.flatten().tolist(),
            'prediction_last': prediction_last.flatten().tolist(),
            'target_all': target_all.flatten().tolist(),
            'target_end': target_end.flatten().tolist(),
            'target_last': target_last.flatten().tolist(),

            'correlation_all': correlation_all.flatten().tolist(),
            'correlation_end': correlation_end.flatten().tolist(),

            'prediction_variance_all': prediction_var_all.flatten().tolist(),
            'prediction_variance_end': prediction_var_end.flatten().tolist(),

            'target_variance_all': target_var_all.flatten().tolist(),
            'target_variance_end': target_var_end.flatten().tolist(),
        }
        # Aggregate statistics
        for key, item in list(res.items()):
            res['%s_mean' % key] = sum(item)/len(item)

        res['cnt_all'] = self._cnt.cnt()
        res['cnt_end'] = self._cnt.cnt_end()

        return res

    @staticmethod
    def average_stats(stats):
        res = {}
        example_cnt = len(stats)

        if example_cnt == 0:
            return res

        cnt_all = sum(s['cnt_all'] for s in stats)
        cnt_end = sum(s['cnt_end'] for s in stats)

        res['example_cnt'] = int(example_cnt)
        res['cnt_all'] = int(cnt_all)
        res['cnt_end'] = int(cnt_end)

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
        for name in ('all', 'end', 'last'):
            dim = len(res['prediction_%s' % name])
            cov = MetricRunningCovariance(dim=dim)

            x = np.concatenate([np.array(s['prediction_%s' % name]) for s in stats]).reshape([example_cnt, dim])
            y = np.concatenate([np.array(s['target_%s' % name]) for s in stats]).reshape([example_cnt, dim])
            cov.append(x, y)

            res['B_correlation_%s' % name] = cov.corr().flatten().tolist()
            res['B_prediction_variance_%s' % name] = cov.var_x().flatten().tolist()
            res['B_target_variance_%s' % name] = cov.var_y().flatten().tolist()
            res['B_l2_loss_%s' % name] = np.mean((x-y)**2)
            res['B_l1_loss_%s' % name] = np.mean(np.abs(x-y))

        return res


class MetricsRegression(MetricsBase):
    def __init__(self, name, output_size, skip_first_cnt):
        super().__init__(Example, name, output_size, skip_first_cnt)



