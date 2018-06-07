from src.data_reading import BaseDataReader
from datetime import datetime
import logging
import os


# Initialize logging
logger = logging.getLogger(__name__)


# Connects all components: Model, Reader, Trainer and runs on training/validation/test data.
# Used for single experiments as well as distributed experiments with Hyperparameter Optimization.
class TrainManager:
    def __init__(self, ModelClass, ReaderClass, TrainerClass, working_dir, validation_data_type, **kwargs):
        self.ModelClass = ModelClass
        self.ReaderClass = ReaderClass
        self.TrainerClass = TrainerClass
        self.working_dir = working_dir
        self.validation_data_type = validation_data_type

    @staticmethod
    def add_arguments(parser):
        parser.section('train_manager')
        parser.add_argument("run_log_folder", type=str, default='',
                            help="Folder used to log training results and store the model")
        parser.add_argument("validation_data_type", type=str, default=BaseDataReader.Validation_Data,
                            choices=BaseDataReader.DataTypes,
                            help="What data type will be used for validation")

    # Create new directory for this run
    @staticmethod
    def get_unique_dir():
        return datetime.utcnow().strftime('%Y_%m_%d__%H_%M_%S_%f')

    def train(self, experiment_arguments):
        log_dir = self._log_dir(experiment_arguments)

        model = self._initialize_model(experiment_arguments)

        # Main code that trains the model for a given budget
        train_metrics = self._run(model, experiment_arguments, BaseDataReader.Train_Data, train=True)

        # Save results, config file and the model; we will be able to recover everything.
        experiment_arguments.save_to_file(file_path=os.path.join(log_dir, 'config.ini'))
        train_metrics.save(directory=log_dir)
        model.save_model(os.path.join(log_dir, 'model'))

        return train_metrics

    def validate(self, experiment_arguments, save_metrics=True):
        model = self._initialize_model(experiment_arguments)

        valid_metrics = self._run(model, experiment_arguments, data_type=self.validation_data_type, train=False)

        if save_metrics:
            # Save validation/test results in a separate folder (we might want to store multiple validation results)
            valid_metrics.save(directory=os.path.join(self._log_dir(experiment_arguments), self.get_unique_dir()))

        return valid_metrics

    def _run(self, model, experiment_arguments, data_type=BaseDataReader.Train_Data, train=False):
        args = experiment_arguments.get_arguments()

        if data_type is not BaseDataReader.Train_Data and train is True:
            raise RuntimeError('You try to train the network using validation or test data!')

        if data_type is BaseDataReader.Train_Data and train is False:
            logger.warning('You use training data but you do not train the network.')

        if data_type == 'train' or experiment_arguments.validation_sequence_size == 0:
            sequence_size = experiment_arguments.sequence_size
        else:
            sequence_size = experiment_arguments.validation_sequence_size

        offset_size = model.offset_size(sequence_size=sequence_size)
        logger.info('Data reader will use an offset:  %d' % offset_size)

        allow_smaller_batch = False if train else True
        dr = self.ReaderClass(offset_size=offset_size, allow_smaller_batch=allow_smaller_batch,
                              state_initializer=model.initial_state, data_type=data_type, **args)

        trainer = self.TrainerClass(model=model, **args)

        metrics = trainer.run(data_reader=dr, train=train)

        return metrics

    def _initialize_model(self, experiment_arguments):
        args = experiment_arguments.get_arguments()
        context_size = self.ReaderClass.context_size(**args)
        input_size = self.ReaderClass.input_size(**args)
        output_size = self.ReaderClass.output_size(**args)

        model = self.ModelClass(input_size=input_size, output_size=output_size, context_size=context_size, **args)

        # Try to restore the model
        model_path = os.path.join(self._log_dir(experiment_arguments), 'model')
        try:
            model.load_model(model_path)
            logger.info('Model loaded from %s' % model_path)
        except FileNotFoundError:
            logger.info('New model created in %s' % model_path)

        logger.info('Number of parameters in the model %d' % model.count_params())
        return model

    def _log_dir(self, experiment_arguments):
        assert experiment_arguments.run_log_folder is not '', 'Forgot to set run_log_folder ?'
        log_dir = os.path.join(self.working_dir, 'train_manager', experiment_arguments.run_log_folder)
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
