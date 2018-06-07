from src.result_processing.metrics_base import MetricsBase, MetricsExampleBase
from src.result_processing.utils import MetricSum, MetricCount, MetricSingleLabelAccuracy, MetricAllEndLast
import numpy as np


class Example(MetricsExampleBase):
    """
    Class used to compute statistics for single recording. If recording is long it might be divided into many
    chunks, data chunks might be even overlapping for train data if random_mode = 2 or if continuous = 1.
    Note: The way we compute accuracy using the sum of log probabilities over the whole recording works only
    because we have one label per recording. Take into account that you will not be able to use this
    class for the dataset that does not fulfil this requirement.
    """
    def __init__(self, example_id, output_size, skip_first_cnt):
        super().__init__(example_id=example_id, output_size=output_size, skip_first_cnt=skip_first_cnt)
        # Has to be deduced after receiving first results
        self.true_label = None

        self._cnt = MetricCount(dim=output_size)
        self._acc = MetricAllEndLast(metric_class=MetricSingleLabelAccuracy, dim=output_size)
        self._loss = MetricAllEndLast(metric_class=MetricSum, dim=output_size)

    # Output should have shape [timepoints x labels_cnt]
    def _append(self, output, labels):
        self.true_label = self.true_label if self.true_label is not None else labels[0]
        if not all(x == self.true_label for x in labels):
            print(labels)
            print(self.true_label)
        assert all(x == self.true_label for x in labels), 'All labels should have the same value'

        # Set a minimum of 1%
        output = np.clip(output, -4.61, 0)

        self._cnt.append(x=output, y=labels)
        self._acc.append(x=output, y=labels)
        self._loss.append(x=output, y=labels)

    def stats(self):
        loss_all, loss_end, loss_last = [r.flatten()[self.true_label] for r in self._loss.mean_x()]
        vote_acc_all, vote_acc_end, vote_acc_last = self._acc.vote_accuracy()
        vote_prob_all, vote_prob_end, vote_prob_last = self._acc.vote_probability()
        log_acc_all, log_acc_end, log_acc_last = self._acc.log_accuracy()
        log_prob_all, log_prob_end, log_prob_last = self._acc.log_probability()

        res = {
            'true_label': int(self.true_label),
            'cnt_all': int(self._cnt.cnt()),
            'cnt_end': int(self._cnt.cnt_end()),

            'loss_all': loss_all,
            'loss_end': loss_end,
            'loss_last': loss_last,

            'vote_acc_all': vote_acc_all,
            'vote_acc_end': vote_acc_end,
            'vote_acc_last': vote_acc_last,

            'vote_prob_all': vote_prob_all.flatten().tolist(),
            'vote_prob_end': vote_prob_end.flatten().tolist(),
            'vote_prob_last': vote_prob_last.flatten().tolist(),

            'log_acc_all': log_acc_all,
            'log_acc_end': log_acc_end,
            'log_acc_last': log_acc_last,

            'log_prob_all': log_prob_all.flatten().tolist(),
            'log_prob_end': log_prob_end.flatten().tolist(),
            'log_prob_last': log_prob_last.flatten().tolist()
        }

        return res

    @staticmethod
    def average_stats(stats):
        """
        Takes statistics over multiple examples and creates one single aggregated description.
        """
        res = {}

        if len(stats) == 0:
            return res

        example_cnt = len(stats)
        cnt_all = sum(s['cnt_all'] for s in stats)
        cnt_end = sum(s['cnt_end'] for s in stats)

        res['example_cnt'] = int(example_cnt)
        res['cnt_all'] = int(cnt_all)
        res['cnt_end'] = int(cnt_end)

        for key in stats[0].keys():
            if any(word in key for word in ['loss', 'vote_acc', 'log_acc']):
                try:
                    res[key] = sum([s[key] for s in stats]) / example_cnt
                except TypeError:
                    res[key] = [sum(b)/example_cnt for b in zip(*[s[key] for s in stats])]
        return res


class MetricsClassification(MetricsBase):
    """
    For normal/abnormal EEG dataset we have very long recordings and we might want to average predictions
    from the full recording instead of taking just the last one.
    """
    def __init__(self, name, output_size, skip_first_cnt):
        super().__init__(Example, name=name, output_size=output_size, skip_first_cnt=skip_first_cnt)


if __name__ == '__main__':
    import json
    dim = 5
    timepoints = 100
    metrics_classification = MetricsClassification(name='test', output_size=dim, skip_first_cnt=0)

    x = np.random.uniform(0, 10, [1, timepoints, dim])
    y = np.zeros(shape=[1, timepoints]).astype(np.int32)
    print(x)
    print(y)
    metrics_classification.append_results(ids=[0], output=x, labels=y, loss=5)

    print(json.dumps(metrics_classification.get_detailed_results(), sort_keys=True, indent=2))
    print(json.dumps(metrics_classification.get_summarized_results(), sort_keys=True, indent=2))

