from src.deep_learning.pytorch.optimizer import ExtendedAdam, CosineScheduler, ScheduledOptimizer
from src.deep_learning.model_trainer import ModelTrainerBase
from src.result_processing import MetricsSimpleLoss
from torch.autograd import Variable
from torch import nn
import logging
import torch
from torch.nn.functional import log_softmax

logger = logging.getLogger(__name__)


criterion_dict = {
    "CrossEntropy": nn.NLLLoss,
    "MeanSquaredError": nn.MSELoss,
    "L1Loss": nn.L1Loss
}


class ModelTrainer(ModelTrainerBase):
    def __init__(self, cuda, **kwargs):
        super().__init__(**kwargs)

        self.cuda = cuda
        self.criterion = criterion_dict[self.objective_type.split('_')[0]]()

        if self.cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        if self.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9,
                                             weight_decay=self.weight_decay)

        elif self.optimizer == "ExtendedAdam":
            logger.info("Will use Extended Adam with weight_decay %s l2_decay %s" % (self.weight_decay, self.l2_decay))
            self.optimizer = ExtendedAdam(self.model.parameters(), lr=self.learning_rate,
                                          weight_decay=self.weight_decay, l2_decay=self.l2_decay)
        else:
            raise NotImplementedError('This optimizer %s is not implemented' % self.optimizer)

        if self.cosine_decay:
            logger.info('Will use cosine restarts decay')
            self.scheduler = CosineScheduler(self.optimizer)
        else:
            self.scheduler = None

        self.optimizer = ScheduledOptimizer(self.optimizer, self.scheduler)

    @staticmethod
    def add_arguments(parser):
        ModelTrainerBase.add_arguments(parser)
        parser.add_argument("cuda", dest="cuda", type=int, default=1, choices=[0, 1],
                            help="Use cuda implementation")

        return parser

    def _one_iteration(self, batch, time, hidden, labels, context, update=False, progress=0.0):
        if update:
            self.model.train()
        else:
            self.model.eval()

        # Forward pass
        batch = Variable(torch.from_numpy(batch))
        labels = Variable(torch.from_numpy(labels))

        if None not in context:
            context = Variable(torch.from_numpy(context))

        if self.cuda:
            batch = batch.cuda()
            labels = labels.cuda()

            # Hidden is a tuple where each element contains a list of states per layer
            # For GRU this tuple has 1 element, for LSTM this tuple has 2 elements
            try:
                if self.model.state_tuple_dim == 1:
                    hidden = [h.cuda() for h in hidden]
                else:
                    hidden = tuple([h.cuda() for h in hidden_ti] for hidden_ti in hidden)
            except AttributeError:
                pass

            if self.model.context_size > 0:
                context = context.cuda()

        batch = self.model.lasso_module(batch)
        outputs, hidden = self.model(batch, hidden, context)

        if 'CrossEntropy' in self.objective_type:
            outputs = log_softmax(outputs, dim=-1)

        if '_last' in self.objective_type:
            training_outputs = outputs[:, -1, :]
            training_labels = labels[:, -1]
        elif '_all' in self.objective_type:
            outputs_num = outputs.size()[-1]
            training_outputs = outputs.view(-1, outputs_num)
            training_labels = labels.view(-1)
        else:
            raise NotImplementedError

        loss = self.criterion(training_outputs, training_labels)
        loss = loss + self.model.lasso_module.loss()

        # Backward pass
        if update:
            self.optimizer.zero_grad()
            loss.backward()

            # total_norm = 0
            # for p in list(filter(lambda p: p.grad is not None, self.model.parameters())):
            #     param_norm = p.grad.data.norm(2)
            #     total_norm += param_norm ** 2
            # total_norm = total_norm ** (1. / 2)
            # print(total_norm)

            if self.gradient_clip != 0:
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.gradient_clip)
            self.optimizer.step(progress=progress)

        return outputs, hidden, loss

    def _gather_results(self, ids, outputs, labels, loss, metrics):
        if self.metrics_class == MetricsSimpleLoss:
            # Save the time by not copying outputs and labels to the CPU
            metrics.append_results(ids, None, None, loss.cpu().data.numpy()[0])
        else:
            metrics.append_results(ids, outputs.cpu().data.numpy(), labels, loss.cpu().data.numpy()[0])

