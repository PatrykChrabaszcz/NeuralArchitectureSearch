from src.experiment_arguments import ExperimentArguments
from src.hpbandster.config_generator import ConfigGenerator
from src.hpbandster.results_logger import ResultLogger
from hpbandster.iterations.successivehalving import SuccessiveHalving
from hpbandster.distributed.utils import start_local_nameserver
from hpbandster.core.master import Master
from threading import Thread
import numpy as np
import logging
import json
import os


# Initialize logging
logger = logging.getLogger(__name__)
master_logger = logging.getLogger('Master')
dispatcher_logger = logging.getLogger('Dispatcher')


class BayesianOptimizer(Master):
    """
    This class directly uses HpBandSter library for join Architecture and Hyperparameter Network optimization.
    It creates Config Generator object that builds a model and is used to generate new configurations.
    For better efficiency HyperBand like evaluation is used.
    """
    @staticmethod
    def add_arguments(parser):
        parser.section('bayesian_optimizer')
        parser.add_argument("config_space_file", type=str, default='',
                            help="File with the configuration space used for Architecture Search (.pcs format).")
        parser.add_argument("n_iterations", type=int, default=100,
                            help="Number of Hyperband iterations for this experiment.")
        parser.add_argument("run_id", type=str, default='0',
                            help="Id of the run")
        parser.add_argument("eta", type=int, default=3,
                            help="In each SequentialHalving iteration 1/eta solutions advance to the next round.")
        parser.add_argument("min_budget", type=float, default=1,
                            help="Budget for the first cycle of HyperBand. Multiplied by eta for each new cycle")
        parser.add_argument("max_budget", type=int, default=81,
                            help="Budget for the last cycle of HyperBand")
        parser.add_argument("ping_interval", type=int, default=10,
                            help="HPBandSter parameter.")
        parser.add_argument("nic_name", type=str, default='eth0',
                            help="Network interface card used for Pyro4.")
        parser.add_argument("bo_loss", type=str, default='',
                            help="Name of the field for bayesian optimizer optimization")
        parser.add_argument("bo_loss_type", type=str, default='minimize', choices=['minimize', 'maximize'],
                            help="Whether to minimize or to maximize bo_loss value")
        return parser

    def __init__(self, working_dir, config_space_file, n_iterations, run_id, eta, min_budget, max_budget, ping_interval,
                 nic_name, **kwargs):

        # Class that is used by the HpBandSter Master to store and manage job configurations and results
        # At the beginning will load results from the previous HpBandSter runs. Those can be used to initialize
        # config_generator

        self.results_logger = ResultLogger(working_dir)

        # Config space that holds all hyperparameters, default values and possible ranges
        self.config_space = ExperimentArguments.read_configuration_space(config_space_file)

        # Config generator that builds a model and samples promising configurations.
        # Initialized from previous configurations if those are present
        self.config_generator = ConfigGenerator(self.config_space, working_dir=working_dir, **kwargs)
        self.config_generator.load_from_results_logger(self.results_logger)

        self.pyro_conf_file = os.path.join(working_dir, 'pyro.conf')

        ns_host, ns_port = start_local_nameserver(nic_name=nic_name)
        logger.info('Started nameserver with %s %s' % (ns_host, ns_port))

        if os.path.exists(self.pyro_conf_file):
            raise RuntimeError('Pyro conf file already exists %s' % self.pyro_conf_file)

        with open(self.pyro_conf_file, 'w') as f:
            logger.info('Creating new Pyro conf file.')
            json.dump({'ns_host': ns_host, 'ns_port': ns_port}, f, sort_keys=True, indent=2)

        super().__init__(run_id=run_id,
                         config_generator=self.config_generator,
                         working_directory=working_dir,
                         ping_interval=ping_interval,
                         nameserver=ns_host,
                         nameserver_port=ns_port,
                         host=ns_host,
                         logger=master_logger,
                         result_logger=self.results_logger)

        # Ugly, but no other way to set it
        self.dispatcher.logger = dispatcher_logger

        # Hyperband related stuff
        self.eta = eta
        self.min_budget = min_budget
        self.max_budget = max_budget

        # Precompute some HB stuff
        self.max_SH_iter = -int(np.log(min_budget / max_budget) / np.log(eta)) + 1
        self.budgets = max_budget * np.power(eta, -np.linspace(self.max_SH_iter - 1, 0, self.max_SH_iter))

        self.config.update({
            'eta': eta,
            'min_budget': min_budget,
            'max_budget': max_budget,
            'budgets': self.budgets,
            'max_SH_iter': self.max_SH_iter,
            'min_points_in_model': self.config_generator.min_points_in_model,
            'top_n_percent': self.config_generator.top_n_percent,
            'num_samples': self.config_generator.num_samples,
            'random_fraction': self.config_generator.random_fraction,
            'bandwidth_factor': self.config_generator.bw_factor,
            'min_bandwidth': self.config_generator.min_bandwidth
        })

        self.n_iterations = n_iterations
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.eta = eta
        self.ping_interval = ping_interval
        self.run_id = run_id
        self.nic_name = nic_name

    def run_in_thread(self):
        # Start optimizer in a separate thread
        thread = Thread(target=self.run, name='Optimizer thread',
                        kwargs={'n_iterations': self.n_iterations})
        thread.daemon = True
        thread.start()
        logger.info('Bayesian Optimizer started')

    def clean_pyro_file(self):
        """
        Removes the pyro conf file from working_dir when finished or interrupted. This is required if we
        want to restore the training in the future without the need to manually delete that file.
        """
        logger.info('Removing Pyro Conf File')
        os.remove(self.pyro_conf_file)

    def get_next_iteration(self, iteration, iteration_kwargs=None):
        """
        BO-HB uses (just like Hyperband) SuccessiveHalving for each iteration.
        See Li et al. (2016) for reference.

        Args:
            iteration: Index of the iteration to be instantiated
            iteration_kwargs: Not used, kept for compatibility
        Returns:
            SuccessiveHalving iteration with the corresponding number of configurations
        """

        # number of 'SH rungs'
        s = self.max_SH_iter - 1 - (iteration % self.max_SH_iter)
        # number of configurations in that bracket
        n0 = int(np.floor(self.max_SH_iter / (s + 1)) * self.eta ** s)
        ns = [max(int(n0 * (self.eta ** (-i))), 1) for i in range(s + 1)]

        logger.info('Get a new Successive Halving Iteration')
        sh = SuccessiveHalving(HPB_iter=iteration,
                               num_configs=ns,
                               budgets=self.budgets[(-s - 1):],
                               config_sampler=self.config_generator.get_config,
                               result_logger=self.results_logger)
        return sh
