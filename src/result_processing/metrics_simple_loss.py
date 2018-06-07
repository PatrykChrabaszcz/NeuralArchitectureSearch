from src.result_processing.metrics_base import MetricsBase


class MetricsSimpleLoss(MetricsBase):
    """This class implements simple metrics that will only compute the loss. For world language models, number
    of classes (words in the dictionary) is huge. Efficient computation of statistics would be required (otherwise
    most of the training would be spend on computing statistics), this is not implemented right now.
    This class bypasses the problem by only storing the loss as computed by the network. Therefore it is faster
    but at the same time less informative."""
    def __init__(self, name, output_size, skip_first_cnt):
        super().__init__(None, name, output_size)
        self.name = name
        self.loss = 0
        self.loss_cnt = 0

        self.recent_loss_array = [0] * 100
        self.recent_loss_bs_array = [0] * 100
        self.recent_loss_index = 0

        self.name = name

    def append_results(self, ids, output, labels, loss):
        batch_size = len(ids)
        self.loss += loss * batch_size
        self.loss_cnt += batch_size

        self.recent_loss_array[self.recent_loss_index] = loss * batch_size
        self.recent_loss_bs_array[self.recent_loss_index] = batch_size
        self.recent_loss_index = (self.recent_loss_index + 1) % 100

        # Here we removed the time consuming part from the base class
        # ...
