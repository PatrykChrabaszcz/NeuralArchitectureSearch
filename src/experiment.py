from src.deep_learning.pytorch.model_trainer import ModelTrainer as TrainerClass
from src.hpbandster.bayesian_optimizer import BayesianOptimizer
from src.hpbandster.budget_decoder import SimpleBudgetDecoder
from src.hpbandster.config_generator import ConfigGenerator
from src.experiment_arguments import ExperimentArguments
from src.deep_learning.train_manager import TrainManager
import src.deep_learning.pytorch.models as model_module
import src.data_reading as reader_module
import src.hpbandster.budget_decoder
from src.utils import setup_logging
import logging
import socket

# Initialize logging
logger = logging.getLogger(__name__)


class Experiment:
    """
    Current design:
        User provides the names of the classes that should be used for reading the data, decoding the hyperband budget,
        etc.. This way each experiment can be saved and restored just based on the .ini file that is saved together
        with the logs.
    Drawback:
        Hard to use it as an external library, since new readers, models etc. have to be placed in a correct directory.
    Solution:
        We could add modules from which imports are made as an arguments to this class constructor.
        We leave it like this for now.
    """
    @staticmethod
    # Parsing arguments
    def add_arguments(parser):
        parser.section('experiment')
        parser.add_argument("working_dir", type=str,
                            help="Directory for results and other important stuff.")
        parser.add_argument("model_class_name", default='SimpleRNN',
                            help="Model class used for the training.")
        parser.add_argument("reader_class_name", default='AnomalyDataReader',
                            help="Reader class used for the training.")
        parser.add_argument("budget_decoder_class_name", default='SimpleBudgetDecoder',
                            help="Class used to update setting based on higher budget.")
        parser.add_argument("backend", default="Pytorch",
                            help="Whether to use Tensorflow or Pytorch.")
        parser.add_argument("verbose", type=int, default=0, choices=[0, 1],
                            help="If set to 1 then log debug messages.")
        parser.add_argument("is_master", type=int, default=0, choices=[0, 1],
                            help="If set to 1 then it will run thread for BO optimization.")
        return parser

    def __init__(self):
        # Parse initial experiment arguments
        initial_arguments = ExperimentArguments(sections=('experiment',), use_all_cli_args=False)
        initial_arguments.add_class_arguments(Experiment)
        initial_arguments.get_arguments()

        self.is_master = initial_arguments.is_master

        self.initialize_backend(initial_arguments.backend.title())

        self.ModelClass = getattr(model_module, initial_arguments.model_class_name)
        self.ReaderClass = getattr(reader_module, initial_arguments.reader_class_name)
        self.BudgetDecoderClass = getattr(src.hpbandster.budget_decoder, initial_arguments.budget_decoder_class_name)
        self.TrainerClass = TrainerClass

        # Populate experiment arguments with arguments from specific classes
        self.experiment_arguments = ExperimentArguments(use_all_cli_args=True)
        self.experiment_arguments.add_class_arguments(Experiment)
        self.experiment_arguments.add_class_arguments(self.ModelClass)
        self.experiment_arguments.add_class_arguments(self.ReaderClass)
        self.experiment_arguments.add_class_arguments(self.TrainerClass)
        self.experiment_arguments.add_class_arguments(TrainManager)
        self.experiment_arguments.add_class_arguments(BayesianOptimizer)
        self.experiment_arguments.add_class_arguments(ConfigGenerator)
        self.experiment_arguments.get_arguments()

        verbose = initial_arguments.verbose
        setup_logging(self.experiment_arguments.working_dir, logging.DEBUG if verbose else logging.INFO)

        self.train_manager = TrainManager(ModelClass=self.ModelClass, ReaderClass=self.ReaderClass,
                                          TrainerClass=self.TrainerClass,
                                          **self.experiment_arguments.get_arguments())
        logger.info('Initialized experiment on %s' % socket.gethostname())
        self.budget_decoder = self.BudgetDecoderClass(**self.experiment_arguments.get_arguments())

    # We might implement in the future backends for different libraries, for example Tensorflow
    @staticmethod
    def initialize_backend(backend):
        assert backend == 'Pytorch', 'Currently only Pytorch backend is implemented'
