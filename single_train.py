from src.experiment import Experiment
import logging
import json


logger = logging.getLogger(__name__)


def main(continuous=False):
    # Experiment will read all arguments from the .ini file and command line interface (CLI).
    experiment = Experiment()

    train_manager = experiment.train_manager
    experiment_arguments = experiment.experiment_arguments

    # Get important arguments
    run_log_folder = experiment_arguments.run_log_folder

    # If run_log_folder is specified:
    #   a) If it is empty then train from scratch
    #   b) If it contains already trained model then will try to load it from that directory.
    # If run_log_folder is not specified initialize a new one and train from scratch
    if run_log_folder == "":
        experiment_arguments.run_log_folder = train_manager.get_unique_dir()

    while True:
        # Train and save the model
        train_metrics = train_manager.train(experiment_arguments)
        # Validate and save the results
        valid_metrics = train_manager.validate(experiment_arguments)

        # Print for the user
        logger.info('Train Metrics:')
        logger.info(json.dumps(train_metrics.get_summarized_results(), indent=2, sort_keys=True))
        logger.info('%s Metrics:' % experiment_arguments.validation_data_type.title())
        logger.info(json.dumps(valid_metrics.get_summarized_results(), indent=2, sort_keys=True))

        if not continuous:
            break


if __name__ == '__main__':
    logger.info('Start Single Train')
    main(continuous=False)
    logger.info('Script successfully finished...')
