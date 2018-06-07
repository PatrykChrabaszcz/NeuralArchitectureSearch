from ConfigSpace.read_and_write.pcs_new import read as read_pcs
from configparser import ConfigParser
from argparse import ArgumentParser
from copy import deepcopy
import logging
import json
import os
import io


# Initialize logging
logger = logging.getLogger(__name__)


class ExperimentArguments(object):
    """
    Class that extracts arguments declared by different classes used for the experiment run.
    Priority in which arguments are assigned, from the lowest:
    1. Default parameters specified in class declaration.
    2. Parameters provided by the user in the .ini file
    3. Parameters provided by the user as CLI arguments
    4. Parameters specified by the Architecture Optimizer.

    Parsing arguments requires couple stages and might look complicated.
    This class makes it possible to achieve our goals:
        - First we need to extract the location of .ini file from the CLI arguments
        - Using .ini file we need to overwrite default script arguments
        - Based on a subset of script arguments we determine which subclasses are used for the experiment.
          For example which DataReader is used.
        - Then we add arguments required by those classes to the parser and again process all CLI arguments
          asserting that all options were recognized and are correct.

    Args:
        sections: If not None then only specified .ini file sections will be parsed
        use_all_cli_args: If true then will assert that all CLI arguments were processed successfully.
            This detects situations when user thinks that given argument will be used but it is not.
    """

    class ExperimentArgumentsParser:
        """
        Extends standard Argument parser by adding sections. When adding new arguments, we need to set section parameter
        first. For example, when adding DataReader arguments, section should be set to "data_reader".
        Parsed arguments will be returned as a dictionary args_dict[section][arg_name].
        This class was introduced to provide a functionality required to update the .ini file with BaysianOptimizer
        arguments or user provided or CLI arguments. Updated .ini file is saved together with the logs in the
        working_dir. It can be later used to restore the model and continue training or perform an evaluation.
        Maybe there is a better/standard approach to get the same functionality (?).

        Args:
            use_all_cli_args: If True then it will assert that all CLI arguments were processed successfully.
                If one of our arguments determines what other arguments are available, then this property becomes
                useful. It might be the case that we are not able to process all cli args before we find out
                what arguments are needed. For example, first we might read from .ini for from CLI arguments,
                which subclass is used for the DataReader and then add arguments from that class to the available
                options. During the second pass we will already know all available arguments, and we might want to
                assert that all of the CLI arguments provided by the user are actually available in the script.
                This way we eliminate one source of possible mistakes.
        """
        def __init__(self, use_all_cli_args=True):
            self.use_all_cli_args = use_all_cli_args
            self._parser = ArgumentParser(allow_abbrev=False)
            self.current_section = None

            # To find section based on argument name
            self._name_to_section = {}
            # Dictionary with section as first key and argument name as second key
            self._args = {}

        def section(self, section):
            """
            Sets the current sections. All arguments added after a call to this function will be assigned to a
            given section
            Args:
                section: Name of the section to be used, if None then the next call to add_argument will throw an
                    assertion error.
            """
            self.current_section = section
            if section not in self._args.keys():
                self._args[section] = {}

        def add_argument(self, name, **kwargs):
            """
            Extends ArgumentParser add_argument(...) function by assigning it to the most recently specified section.
            Args:
                name: Name of the argument.
                kwargs: Arguments forwarded to the internal ArgumentParser object
            """
            assert self.current_section is not None, 'You need to specify arguments section'
            assert name not in self._name_to_section.keys(), 'Argument %s already exists' % name
            self._name_to_section[name] = self.current_section
            self._parser.add_argument('--%s' % name, **kwargs)

        def parse(self, unknown_args):
            """
            Extracts CLI arguments from unknown_args. If class object was initialized with use_all_cli_args=True
            then it will make sure that all CLI arguments were declared in the script, otherwise it will throw an
            exception.
            :param unknown_args: CLI args to be processed. Those are returned by the first parser which only finds
                .ini file location.
            """
            args_dict = deepcopy(self._args)
            args, unknown_args = self._parser.parse_known_args(unknown_args)
            if self.use_all_cli_args:
                if len(unknown_args) > 0:
                    logger.error('Unknown CLI arguments %s' % unknown_args)
                    raise RuntimeError('Unknown CLI arguments %s' % unknown_args)

            # Create a dictionary based on arguments
            for arg, value in vars(args).items():
                section = self._name_to_section[arg]
                args_dict[section][arg] = value

            return args_dict

        def update_defaults(self, **kwargs):
            """
            This function is used to update default arguments that are specified in the script (for an example look
            at SequenceDataReader class) with arguments from the .ini file, which have a higher priority.
            Args:
                kwargs: Arguments forwarded to the internal ArgumentParser object
            """
            self._parser.set_defaults(**kwargs)

    def __init__(self, sections=None, use_all_cli_args=True):

        self._sections = sections
        self._ini_conf = None
        self._ini_file_parser = ArgumentParser(allow_abbrev=False)
        self._ini_file_parser.add_argument("--ini_file", type=str, default="",
                                           help="Path to the file with default values "
                                                "for script parameters (.ini format).")

        self._parser = ExperimentArguments.ExperimentArgumentsParser(use_all_cli_args)
        self._arguments = None

    def save_to_file(self, file_path):
        """
        Save internal .ini file with all updates done to it (CLI arguments, ConfigSpace updates)
        to make it possible to restore experiment with parameters used for training.
        Args:
            file_path: path to the output .ini file

        """
        with open(file_path, 'w') as config_file:
            self._ini_conf.write(config_file)

    @staticmethod
    def read_configuration_space(file_path):
        """
        Load ConfigSpace object from .pcs (parameter configuration space) file.
        Throws an exception if file is not available.
        Args:
            file_path: Path to the .pcs file.
        Returns:
            ConfigSpace object representing configuration space.
        """

        with open(file_path, 'r') as f:
            s = f.readlines()
            config_space = read_pcs(s)
        return config_space

    @staticmethod
    def ini_to_json(ini_file):
        ini_conf = ConfigParser()
        ini_conf.read(ini_file)
        json_dict = {}
        for section in ini_conf.sections():
            json_dict = {**json_dict, **dict(ini_conf.items(section))}

        return json_dict

    def add_class_arguments(self, class_type):
        """
        Adds arguments specified in the class 'class_type' to the internal Argument Parser object.
        This function should be called sequentially for all classes used in the experiment.
        """
        if self._arguments is not None:
            raise RuntimeError('Arguments already parsed!')

        self._parser.current_section = None
        class_type.add_arguments(self._parser)

    def get_arguments(self):
        """
        Gets a dictionary with parsed arguments.
        Argument source priority from the lowest:
        1. Defaults defined in the code
        2. Arguments provided in .ini file
        3. CLI arguments provided by the user

        Updates internal .ini object which can be saved using save_to_file() function.

        Returns:
            Dictionary with script arguments, removes section information making it 1 level deep: arg_dict[argument].
        """
        if self._arguments is not None:
            return self._arguments

        # Initialize ConfigParser to manage .ini file
        self._ini_conf = ConfigParser()

        # Get the file path for the ini_file
        args, unknown_args = self._ini_file_parser.parse_known_args()
        ini_file = args.ini_file

        # By default ini file is an empty string which means that we don't use it
        if ini_file != "":
            assert os.path.isfile(ini_file), 'Could not find specified (%s) ini file' % ini_file

            logger.debug('Updating default parameter values from file: %s' % ini_file)
            self._ini_conf.read(ini_file)

            # Filtered sections if needed or all sections from the ini file

            args = self._parser.parse(unknown_args)

            sections = self._sections if self._sections is not None else args.keys()
            sections = set(sections).intersection(set(self._ini_conf.sections()))
            # Assert that arguments from .ini file are in the script
            for section in sections:
                ini_args_dict = dict(self._ini_conf.items(section))
                args_dict = args[section]

                for key in ini_args_dict.keys():
                    if key not in args_dict.keys():
                        raise RuntimeError('Argument %s: %s from .ini file not present '
                                           'in the script.' % (section, key))

                # Replace script defaults with defaults from the ini file
                self._parser.update_defaults(**ini_args_dict)

        # At this point we should have arguments from default values, .ini file values and CLI values
        args = self._parser.parse(unknown_args)

        # Now save everything to the ConfigParser, such that we will be able to save and restore those parameters
        # And flatten nested args such that we will be able to use them as **kwargs
        self._arguments = {}
        for section in args.keys():
            for arg_name, arg_value in args[section].items():
                if section not in self._ini_conf.sections():
                    self._ini_conf.add_section(section)
                self._ini_conf.set(section, arg_name, str(arg_value))
                self._arguments[arg_name] = arg_value

        return self._arguments

    def updated_with_configuration(self, configuration):
        """
        Will copy current parameters and update this copy based on the configuration object.
        Configuration object can come from the Architecture Search optimizer.
        Note:
            Does not modify original parameters! Multiple calls with different configurations will alter original
            parameters.
        Args:
            configuration: Configuration object (from ConfigSpace library).
        Returns:
            Dictionary with updated arguments, same structure as dictionary returned by get_arguments(..).
        """

        arguments = self.copy()
        for arg_name in configuration.keys():
            new_arg_value = type(arguments[arg_name])(configuration[arg_name])
            old_arg_value = arguments[arg_name]
            logger.debug('Changing %s from %s to %s' % (arg_name, old_arg_value, new_arg_value))
            arguments[arg_name] = new_arg_value

        return arguments

    def copy(self):
        """
        Makes a deep copy of already parsed arguments.
        Copies internal _arguments dictionary and .ini file object.
        Returns:
            Deep copy of this object
        """
        assert self._ini_conf is not None and self._arguments is not None, 'Can only copy initialized arguments!'

        experiment_arguments = ExperimentArguments()
        experiment_arguments._arguments = deepcopy(self._arguments)

        config_string = io.StringIO()
        self._ini_conf.write(config_string)
        # We must reset the buffer ready for reading.
        config_string.seek(0)
        experiment_arguments._ini_conf = ConfigParser()
        experiment_arguments._ini_conf.read_file(config_string)

        return experiment_arguments

    # Function below allow direct access to the internal _arguments dictionary. It simplifies the code that
    # uses objects of this class. Note that it also updates internal .ini object whenever any argument is updated.
    def __getattr__(self, item):
        if item[0] == '_':
            return super().__getattribute__(item)
        else:
            return self._arguments[item]

    def __setattr__(self, key, value):
        if key[0] == '_':
            super().__setattr__(key, value)
        else:
            if key not in self._arguments.keys():
                raise KeyError('ExperimentArguments does not support addition of new arguments during runtime. '
                               '(Argument name: %s)' % key)
            self._arguments[key] = value

            # Makes it possible to save again updated by CLI arguments .ini
            # file and restore experiment
            for section in self._ini_conf.sections():
                if self._ini_conf.has_option(section, key):
                    self._ini_conf.set(section, key, str(value))
                    return
            raise RuntimeError('Could not set field %s in the ConfigParser object' % key)

    def __getitem__(self, item):
        return self.__getattr__(item)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)
