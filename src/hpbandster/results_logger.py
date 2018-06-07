from ConfigSpace import Configuration
import numpy as np
import logging
import json
import os
import re

logger = logging.getLogger(__name__)


class ResultLogger(object):
    """
    Class that is used to save explored configurations and corresponding results.
    Called from HpBandSter Master class each time new job is finished and reported.
    Can be used to initialize ConfigurationSpace sampler from the previously done experiments.
    """

    class Configs:
        """
        Class that manages configurations that were sampled in one of the experiment runs.
        We can run the same experiment multiple times and we would like to be able to use information from
        the previous runs. Configurations are stored as dictionaries. Use get_configs to access them as
        Configuration [from ConfigSpace library] objects
        """
        def __init__(self, configs_file):
            self.configs_file = configs_file
            # Dictionary for configurations in a json format. Do not use, use get_configuration() instead
            self._configs = {}
            try:
                with open(self.configs_file) as f:
                    # Each line contains Json Object with a single configuration
                    for line in f.readlines():
                        configuration_id, configuration_params, info = json.loads(line)
                        configuration_id = tuple(configuration_id)
                        assert configuration_id not in list(self._configs.keys()), \
                            'Configuration %s present twice while it should be unique.' % configuration_id
                        self._configs[configuration_id] = [configuration_params, info]
            except IOError:
                # If file is not present we will do not load anything
                pass

        def add(self, config_id, config, config_info):
            """
            After worker finishes his job, the job will be sent to the HpBandSter master module. This will in turn
            call ResultLogger callback to register the job. ResultLogger will use Configs object to store and
            manage evaluated configurations.
            """
            # If configuration is already present (for example it was executed for a lower budget) we do nothing.
            if config_id not in self._configs.keys():
                config = [config, config_info]
                self._configs[config_id] = config

                # Save it to the file
                with open(self.configs_file, 'a') as f:
                    logger.debug('Saving new configuration %s to the configs file %s' % (config_id, self.configs_file))
                    f.write(json.dumps([config_id] + config, sort_keys=True))
                    f.write('\n')

        def get_configuration(self, configuration_id, configuration_space):
            configuration_id = tuple(configuration_id)
            return Configuration(configuration_space=configuration_space, values=self._configs[configuration_id][0])

        def get_configuration_info(self, configuration_id):
            return self._configs[tuple(configuration_id)][1]

    class Results:
        """
        Class that manages results from a given configuration for a given budget. Function add_job will be called
        every time new results arrives from the worker to the HpBandSter master. This function will store the result
        and save it to the log file. It can be initialized with results from the previous experimental run.
        """
        def __init__(self, results_file):
            self.results_file = results_file
            self.results = []

            try:
                with open(results_file) as f:
                    for line in f.readlines():
                        # Read important stuff, ignore for now not the rest
                        result = json.loads(line)
                        self.results.append(result)
            except FileNotFoundError:
                # If file is not present we will do not load anything
                pass

        def add_job(self, job):
            """
            After worker finishes his job, the job will be sent to the HpBandSter master module. This will in turn
            call ResultLogger callback to register the job. ResultLogger will use Results object to store and
            manage obtained results.
            """
            # Extract what was saved in the original HpBandSter implementation
            r = [job.id, job.kwargs['budget'], job.timestamps, job.result, job.exception]
            self.results.append(r)

            with open(self.results_file, 'a') as f:
                logger.debug('Saving new result %s to the results file %s' % (job.id, self.results_file))
                f.write(json.dumps(r, sort_keys=True))
                f.write('\n')

        def get_results(self):
            for r in self.results:
                yield r

    def __init__(self, working_dir):
        self.working_dir = working_dir

        os.makedirs(working_dir, exist_ok=True)

        indices = self._find_indices()
        self.current_index = max(indices) + 1 if indices else 0

        logger.info('We found %d previous run/runs, this run will use previous results and save '
                    'new ones using an index %d' % (len(indices), self.current_index))

        # Loads previous results if there are any
        self.historical_configs = [ResultLogger.Configs(os.path.join(self.working_dir, 'configs_%d.json' % index))
                                   for index in indices]
        self.historical_results = [ResultLogger.Results(os.path.join(self.working_dir, 'results_%d.json' % index))
                                   for index in indices]

        # Used to log new results
        self.current_configs = ResultLogger.Configs(
            os.path.join(self.working_dir, 'configs_%d.json' % self.current_index))
        self.current_results = ResultLogger.Results(
            os.path.join(self.working_dir, 'results_%d.json' % self.current_index))

    def get_results(self, configuration_space, with_info=False):
        """
        Will generate one by one configurations together with loss and budget. Can be used to recover the state of
        ConfigGenerator object after training is restored
        Args:
            configuration_space: Defines how json configuration description is translated into a Configuration object.
            with_info: If set to true then will also return information whether this was model sampled configuration.
        Returns:
            configuration: Configuration parameters for a given train/validation run.
            budget: Budget for which given configuration was executed.
            loss: Reported performance achieved by the model.
        """

        for h_r, h_c in zip(self.historical_results, self.historical_configs):
            for result in h_r.get_results():
                configuration_id = result[0]
                configuration = h_c.get_configuration(configuration_id, configuration_space=configuration_space)
                if with_info:
                    yield configuration, result[1:], h_c.get_configuration_info(configuration_id)
                else:
                    yield configuration, result[1:]

    def _find_indices(self):
        """
        Will look at the current working_dir and check if files matching the pattern 'configs_%d.json" and
        "results_%d.json". For both patterns it will find what numbers are used and assert that each configs file has
        its corresponding results file.

        Returns:
            Sorted indices of files with historical data from the previous runs in this working_dir.
        """
        def __find_indices(name):
            files = [f.name for f in os.scandir(self.working_dir) if name in f.name]
            matched = [re.findall("%s_(\d+).json" % name, file) for file in files]
            matched = [int(v[0]) for v in matched if v]

            return sorted(matched)

        try:
            config_indices, result_indices = [__find_indices(name) for name in ['configs', 'results']]
            assert len(config_indices) == len(result_indices)

            indices = config_indices

            return indices
        except IndexError:
            return []

    def new_config(self, config_id, config, config_info):
        self.current_configs.add(config_id, config, config_info)

    def __call__(self, job):
        """
        Method called by the HpBandSter master module.
        Simply forwards job config and results into the designated objects that will log them.
        """
        self.current_results.add_job(job)
