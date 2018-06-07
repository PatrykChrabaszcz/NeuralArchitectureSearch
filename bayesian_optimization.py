from src.experiment import Experiment
from src.hpbandster.bayesian_optimizer import BayesianOptimizer
from src.hpbandster.worker import Worker
import logging


logger = logging.getLogger(__name__)


# Some thoughts:
# If this node is marked as 'is_master', then we will start Bayesian Optimizer in a separate thread and worker in the
# main thread. This might mean that worker in this process will have less time for training compared to other processes.
# The influence of this should not be big, but it should be checked in the future.
def main():
    # Experiment will read all arguments from the .ini file and command line interface (CLI).
    experiment = Experiment()

    if experiment.is_master:
        # Initialize bayesian optimizer
        optimizer = BayesianOptimizer(**experiment.experiment_arguments.get_arguments())

        try:

            # Start optimizer in a thread
            optimizer.run_in_thread()

            worker = Worker(train_manager=experiment.train_manager,
                            budget_decoder=experiment.budget_decoder,
                            experiment_args=experiment.experiment_arguments,
                            **experiment.experiment_arguments.get_arguments())

            worker.run()
        finally:
            logger.info('Cleaning up optimizer object')
            optimizer.clean_pyro_file()
    else:
        worker = Worker(train_manager=experiment.train_manager,
                        budget_decoder=experiment.budget_decoder,
                        experiment_args=experiment.experiment_arguments,
                        **experiment.experiment_arguments.get_arguments())

        worker.run()


if __name__ == '__main__':
    print('Start Bayesian Optimization')
    main()
    print('Script successfully finished...')
