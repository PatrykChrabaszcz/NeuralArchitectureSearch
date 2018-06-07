from hpbandster.core.worker import Worker as HpBandSterWorker
from hpbandster.api.util import nic_name_to_host
from src.result_processing import MetricsBase
from time import sleep
import logging
import json
import os


# Initialize logging
logger = logging.getLogger(__name__)
hpbandster_logger = logging.getLogger('HPBandSter')
hpbandster_logger.setLevel(logging.DEBUG)


class Worker(HpBandSterWorker):
    def __init__(self, train_manager, budget_decoder, experiment_args, working_dir, nic_name, run_id,
                 bo_loss, bo_loss_type, **kwargs):
        assert bo_loss_type in ['minimize', 'maximize']

        logger.info('Creating worker for distributed computation.')

        self.bo_loss = bo_loss
        self.bo_loss_type = bo_loss_type
        self.train_manager = train_manager
        self.budget_decoder = budget_decoder
        self.experiment_args = experiment_args
        self.working_dir = working_dir

        ns_name, ns_port = self._search_for_name_server()
        logger.info('Worker found nameserver %s, %s' % (ns_name, ns_port))

        host = nic_name_to_host(nic_name)
        logger.info('Worker will try to run on a host %s' % host)

        super().__init__(run_id=run_id, nameserver=ns_name, nameserver_port=ns_port, host=host,
                         logger=hpbandster_logger)

    def compute(self, config, budget, **kwargs):
        logger.info('Worker: Starting computation for budget %s ' % budget)

        if config is None:
            raise RuntimeError('Worker received config that is None in compute(...)')

        adjusted_experiment_args = self.experiment_args.updated_with_configuration(config)
        adjusted_experiment_args = self.budget_decoder.adjusted_arguments(adjusted_experiment_args, budget)

        # Each evaluation can mean multiple folds of CV
        result_list = []
        for experiment_args in adjusted_experiment_args:
            # If we do not restore the training initialize a new directory
            if experiment_args.run_log_folder == "":
                experiment_args.run_log_folder = self.train_manager.get_unique_dir()

            train_metrics = self.train_manager.train(experiment_args)

            valid_metrics = self.train_manager.validate(experiment_args)

            # Print for the user
            logger.info('Train Metrics:')
            logger.info(json.dumps(train_metrics.get_summarized_results(), indent=2, sort_keys=True))
            logger.info('%s Metrics:' % self.train_manager.validation_data_type.title())
            logger.info(json.dumps(valid_metrics.get_summarized_results(), indent=2, sort_keys=True))

            result_list.append(valid_metrics.get_summarized_results())

        averaged_results = MetricsBase.average_metrics_results(result_list)
        loss = -averaged_results[self.bo_loss] if self.bo_loss_type == 'maximize' else averaged_results[self.bo_loss]
        logger.info('Computation done, submit results (loss %s)' % loss)

        return {
            'loss': loss,
            'info': averaged_results
        }

    def _search_for_name_server(self, num_tries=60, interval=1):
        """
        Will try to find pyro.conf file in the current working_dir and extract ns_name and ns_port parameters.
        Will update internal parameters if values for the current experiment were found
        Args:
            num_tries:
            interval:
        """

        conf_file = os.path.join(self.working_dir, 'pyro.conf')

        user_notified = False
        for i in range(num_tries):
            try:
                with open(conf_file, 'r') as f:
                    d = json.load(f)
                logger.debug('Found nameserver info %s' % d)
                return d['ns_host'], d['ns_port']

            except FileNotFoundError:
                if not user_notified:
                    logger.info('Config file not found. Waiting for the master node to start')
                    user_notified = True
                sleep(interval)

        raise RuntimeError("Could not find the nameserver information after %d tries, aborting!" % num_tries)

