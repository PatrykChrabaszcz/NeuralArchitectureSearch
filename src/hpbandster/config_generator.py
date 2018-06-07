from hpbandster.config_generators.base import base_config_generator
from src.hpbandster.results_logger import ResultLogger
from ConfigSpace import Configuration
from ConfigSpace import OrdinalHyperparameter, CategoricalHyperparameter
import traceback
import scipy.stats as sps
import statsmodels.api as sm
import numpy as np
import logging


# Initialize logging
logger = logging.getLogger(__name__)


class ConfigGenerator(base_config_generator):
    """
    Wrapper around BOHB from HpBanSter [https://github.com/automl/HpBandSter]
    We add arguments to the CLI options and log some information each time new result is received.
    This class keeps track of already evaluated configurations for different budgets. When sufficient number of points
    is recorded it will build a Bayesian Optimization model and sample new configurations using that model.
    """

    class OrdinalChecker:
        """
        Ordinal hyperparameters are for some reason treated more like categorical than integer.
        We need to address this problem
        """

        def __init__(self, config_space):
            self.config_space = config_space
            self.indices = []
            self.sizes = []
            for i, h in enumerate(config_space.get_hyperparameters()):
                if isinstance(h, OrdinalHyperparameter):
                    self.indices.append(i)
                    self.sizes.append(len(h.sequence))

        def get_array(self, configuration):
            array = configuration.get_array()

            for i, s in zip(self.indices, self.sizes):
                # v has value 0, 1, 2, 3, ...
                v = array[i]
                # We need to convert it to [0, 1]
                n_v = (v + 0.5) / s

                array[i] = n_v
            return array

        def get_configuration(self, vector):
            vector = np.copy(vector)

            for i, s in zip(self.indices, self.sizes):
                # v has value [0, 1]
                v = vector[i]
                # Convert back to 0, 1, 2, 3, ...
                n_v = v*s - 0.5
                # For example if s is 4 and v in 1 then 3.5 would round to 4 but we want it to round to 3.
                vector[i] = np.clip(round(n_v), 0, s-1)

            return Configuration(self.config_space, vector=vector)

    @staticmethod
    def add_arguments(parser):
        parser.section('config_generator')
        parser.add_argument("min_points_in_model", type=int, default=64,
                            help="Minimal number of points for a given budget that are required to start building a "
                                 "model used for Bayesian Optimization")
        parser.add_argument("top_n_percent", type=int, default=15,
                            help="Percentage of top solutions used as good.")
        parser.add_argument("num_samples", type=int, default=81,
                            help="How many samples are used to optimize EI (Expected Improvement).")
        parser.add_argument("random_fraction", type=float, default=1./3,
                            help="To guarantee exploration, we will draw random samples from the configuration space.")
        parser.add_argument("bandwidth_factor", type=int, default=3,
                            help="HpBandSter samples from wider KDE (Kernel Density Estimator) to keep diversity.")
        parser.add_argument("min_bandwidth", type=float, default=0.001,
                            help="When all good samples have the same value KDE will have bandwidth of 0. "
                                 "Force minimum bandwidth to diversify samples.")
        parser.add_argument("bw_estimation_method", type=str, default='normal_reference',
                            choices=['normal_reference', 'cross_validation'],
                            help="Method for kernel density estimator bandwidth selection.")
        return parser

    def __init__(self, config_space, working_dir, min_points_in_model, top_n_percent, num_samples, random_fraction,
                 bandwidth_factor, min_bandwidth, bw_estimation_method, **kwargs):
        assert 0 < top_n_percent < 100
        assert 0.0 <= random_fraction <= 1.0
        if min_points_in_model < len(config_space.get_hyperparameters()) + 1:
            min_points_in_model = len(config_space.get_hyperparameters()) + 1
            logger.warning('Minimum number of points in the model is too low, will use %d.' % min_points_in_model)

        super().__init__(logger=logger)

        self.config_space = config_space
        self.ordinal_checker = ConfigGenerator.OrdinalChecker(config_space)
        self.working_dir = working_dir
        self.min_points_in_model = min_points_in_model
        self.top_n_percent = top_n_percent
        self.num_samples = num_samples
        self.random_fraction = random_fraction
        self.bw_factor = bandwidth_factor
        self.min_bandwidth = min_bandwidth
        self.bw_estimation_method = bw_estimation_method

        hps = self.config_space.get_hyperparameters()

        self.kde_vartypes = ''.join(['u' if hasattr(h, 'choices') else 'c' for h in hps])
        self.vartypes = np.array([len(h.choices) if hasattr(h, 'choices') else 0 for h in hps], dtype=int)

        # store precomputed probs for the categorical parameters
        self.cat_probs = []

        # First key is a budget, then list with different configurations
        self.configs = dict()
        # First key is a budget, then loss that corresponds to the configuration
        self.losses = dict()

        # Kernel Density Estimator models for good and bad samples
        self.kde_models = dict()

    def load_from_results_logger(self, results_logger):
        """
        Function that can be called at the beginning. It will use results_logger object to access configurations
        and results obtained with previous runs. It is up to the user to make sure that the same configuration space
        was used before.
        Args:
            results_logger: Object that holds information about previous configurations and results
        """

        try:
            for configuration, r in results_logger.get_results(self.config_space):
                budget = r[0]
                loss = np.inf if r[2] is None else r[2]["loss"]
                self.add_configuration(configuration, budget, loss)
        except:
            print('Except')

        # Start from the biggest budget
        for budget in sorted(self.configs.keys())[::-1]:
            if self.update_kde_model(budget) is True:
                logger.debug('Successfully created a model for budget %s' % budget)
                break
            logger.debug('Could not create model for budget %s' % budget)

    def get_config(self, budget):
        """
        Function to sample a new configuration.
        Successive halving iterations in hyperband will use this function to query a new configuration.
        At the beginning when not enough points are observed those will be simply random configurations.
        Later configurations will be sampled from the BO model that was created for the highest budget.
        We assume that it is the most accurate one.
        Args:
            budget: Budget for the training. Higher budget should give better final performance estimation.
            How to interpret the budget depends on particular experiment. See BudgetDecoder class for example.
        Returns:
            Configuration that can be used for the training
        """

        # We always have some chance to sample from random distribution
        # If not enough points seen for any budget then also sample from a random distribution
        if len(self.kde_models.keys()) == 0 or np.random.rand() < self.random_fraction:
            logger.debug('Sample new random configuration.')
            return self.config_space.sample_configuration().get_dictionary(), dict(model_based_pick=False)

        try:
            logger.debug('Try to sample model based configuration.')

            # We assume that highest budget gives the best estimate of the final performance
            budget = max(self.kde_models.keys())

            l = self.kde_models[budget]['good'].pdf
            g = self.kde_models[budget]['bad'].pdf
            bo_objective = lambda x: max(1e-8, g(x)) / max(l(x), 1e-8)

            kde_good = self.kde_models[budget]['good']
            kde_bad = self.kde_models[budget]['bad']

            # We need to run some random search on top of BO model to find which points are the promising ones.
            best = np.inf
            best_proposed_point = None
            for i in range(self.num_samples):
                # We will sample new proposed points close to random good points.
                # I guess it is more efficient.
                good_point = kde_good.data[np.random.randint(0, len(kde_good.data))]
                proposed_point = []

                # Iterate each dimension and sample independently each dimension
                for m, bw, t in zip(good_point, kde_good.bw, self.vartypes):

                    bw = max(bw, self.min_bandwidth)
                    # Continuous -> Gaussian
                    if t == 0:
                        bw = self.bw_factor * bw
                        try:
                            proposed_point.append(sps.truncnorm.rvs(-m / bw, (1 - m) / bw, loc=m, scale=bw))
                        # TODO: Figure out what error is thrown
                        except KeyError:
                            logger.warning("Truncated Normal failed for: "
                                           "\ndatum=%s\nbandwidth=%s\nfor entry with value %s" %
                                           (good_point, kde_good.bw, m))

                            logger.warning("Data in the KDE:\n%s" % kde_good.data)
                    # Categorical -> (1-bw) probability for the current value, bw/(n-1) probability for any other value
                    else:
                        if np.random.rand() < (1 - bw):
                            proposed_point.append(int(m))
                        else:
                            proposed_point.append(np.random.randint(t))

                val = bo_objective(proposed_point)

                # I do not think it triggers
                if not np.isfinite(val):
                    logger.warning('Sampled vector: %s has EI value %s' % (proposed_point, val))
                    logger.warning("Data in the KDEs:\n%s\n%s" % (kde_good.data, kde_bad.data))
                    logger.warning("Bandwidth of the KDEs:\n%s\n%s" % (kde_good.bw, kde_bad.bw))
                    logger.warning("l(x) = %s" % (l(proposed_point)))
                    logger.warning("g(x) = %s" % (g(proposed_point)))

                    # Right now, this happens because a KDE does not contain all values for a categorical parameter
                    # this cannot be fixed with the statsmodels KDE, so for now, we are just going to evaluate this one
                    # if the good_kde has a finite value, i.e. there is no config with that value in the bad kde,
                    # so it shouldn't be terrible.
                    if np.isfinite(l(proposed_point)):
                        best_proposed_point = proposed_point
                        break

                if val < best:
                    best = val
                    best_proposed_point = proposed_point

            if best_proposed_point is None:
                logger.debug("Sampling based optimization with %i samples failed."
                             "Using random configuration" % self.num_samples)
                sample = self.config_space.sample_configuration()
                info_dict = dict(model_based_pick=False)
            else:
                logger.debug('Best point proposed to evaluate: {}, {}'.format(best_proposed_point, best))
                sample = self.ordinal_checker.get_configuration(vector=best_proposed_point)
                info_dict = dict(model_based_pick=True)
        except KeyError:
            self.logger.warning("Sampling based optimization with %i samples failed\n %s \nUsing random configuration" %
                                (self.num_samples, traceback.format_exc()))
            sample = self.config_space.sample_configuration()
            info_dict = dict(model_based_pick=False)

        return sample.get_dictionary(), info_dict

    def impute_conditional_data(self, array):
        return_array = np.empty_like(array)
        for i in range(array.shape[0]):
            datum = np.copy(array[i])
            nan_indices = np.argwhere(np.isnan(datum)).flatten()

            while np.any(nan_indices):
                nan_idx = nan_indices[0]
                valid_indices = np.argwhere(np.isfinite(array[:, nan_idx])).flatten()

                if len(valid_indices) > 0:
                    # pick one of them at random and overwrite all NaN values
                    row_idx = np.random.choice(valid_indices)
                    datum[nan_indices] = array[row_idx, nan_indices]

                else:
                    # no good point in the data has this value activated, so fill it with a valid but random value
                    t = self.vartypes[nan_idx]
                    if t == 0:
                        datum[nan_idx] = np.random.rand()
                    else:
                        datum[nan_idx] = np.random.randint(t)

                nan_indices = np.argwhere(np.isnan(datum)).flatten()
            return_array[i, :] = datum
        return return_array

    def new_result(self, job):
        """
        Function used to register finished runs. It will be called by HpBandSter master module.
        Args:
            job: Job object containing all the info about the run.
        """

        configuration, budget, loss = self.extract_configuration(job)
        self.add_configuration(configuration=configuration, budget=budget, loss=loss)

        # If we already have a model for a higher budget we will not use information from this run
        # We assume that models with higher budgets are more accurate and use them instead.
        if max(list(self.kde_models.keys()) + [-np.inf]) <= budget:
            self.update_kde_model(budget)

    def extract_configuration(self, job):
        # One could skip crashed results, but we decided to assign a +inf loss
        # We count them as bad configurations
        if job.result is None:
            logger.warning("Job %s failed with exception\n%s".format(job.id, job.exception))
            loss = np.inf
        else:
            loss = job.result["loss"]

        budget = job.kwargs["budget"]

        # We want to get a numerical representation of the configuration in the original space
        configuration = Configuration(self.config_space, job.kwargs["config"])

        return configuration, budget, loss

    def add_configuration(self, configuration, budget, loss):
        if budget not in self.configs.keys():
            self.configs[budget] = []
            self.losses[budget] = []

        # Save new results
        # Standardize values (categorical: {0, 1, ...} integer, ordinal and continuous [0, 1])
        self.configs[budget].append(self.ordinal_checker.get_array(configuration))
        self.losses[budget].append(loss)

    def update_kde_model(self, budget):
        if len(self.configs[budget]) <= self.min_points_in_model + 1:
            logger.debug("Only %i run(s) for budget %f available, need more than %s. Can't build the model!" %
                         (len(self.configs[budget]), budget, self.min_points_in_model + 1))
            return False

        train_configs = np.array(self.configs[budget])
        train_losses = np.array(self.losses[budget])
        sorted_train_configs = train_configs[np.argsort(train_losses)]

        sample_cnt = train_configs.shape[0]

        n_good = max(self.min_points_in_model, (self.top_n_percent * sample_cnt) // 100)
        n_bad = max(self.min_points_in_model, ((100 - self.top_n_percent) * sample_cnt) // 100)

        # Low loss -> Good data points
        train_data_good = self.impute_conditional_data(sorted_train_configs[:n_good])
        train_data_bad = self.impute_conditional_data(sorted_train_configs[-n_bad:])

        assert train_data_good.shape[0] > train_data_good.shape[1], 'Number of points lower than number of dimensions!'
        assert train_data_bad.shape[0] > train_data_bad.shape[1], 'Number of points lower than number of dimensions!'

        bad_kde = sm.nonparametric.KDEMultivariate(data=train_data_bad, var_type=self.kde_vartypes,
                                                   bw=self.bw_estimation_method)
        good_kde = sm.nonparametric.KDEMultivariate(data=train_data_good, var_type=self.kde_vartypes,
                                                    bw=self.bw_estimation_method)

        # Apply minimum bandwidth
        bad_kde.bw = np.clip(bad_kde.bw, self.min_bandwidth, None)
        good_kde.bw = np.clip(good_kde.bw, self.min_bandwidth, None)

        # Update models for this budget
        self.kde_models[budget] = dict(good=good_kde, bad=bad_kde)

        logger.debug('Build new model for budget %f based on %i/%i split.' % (budget, n_good, n_bad))
        logger.debug('Best loss for this budget:%f' % (np.min(train_losses)))
        return True


if __name__ == '__main__':
    from src.experiment_arguments import ExperimentArguments

    logger.setLevel(logging.DEBUG)
    working_dir = '/mhome/chrabasp/EEG_Results/BO_Anomaly_6'
    config_space_file = '/mhome/chrabasp/Workspace/EEG/config/anomaly_simple.pcs'
    results_logger = ResultLogger(working_dir=working_dir)
    config_space = ExperimentArguments.read_configuration_space(config_space_file)
    config_generator = ConfigGenerator(config_space=config_space,
                                       working_dir=working_dir,
                                       min_points_in_model=0,
                                       top_n_percent=20,
                                       num_samples=100,
                                       random_fraction=0.3,
                                       bandwidth_factor=3.0,
                                       min_bandwidth=0.001,
                                       bw_estimation_method='normal_reference')

    config_generator.load_from_results_logger(results_logger)

    model = config_generator.kde_models[27.0]['good']
    model_bad = config_generator.kde_models[27.0]['bad']
    # print(model)
    # print(model_bad)
    #
    print(model.data[:, 8])
    for i, h, bw, t in zip(range(len(model.bw)), config_space.get_hyperparameters(), model.bw, config_generator.vartypes):
        print(h)
        print(t, bw)

        try:
            point = np.array([0.] * 18)
            point[i] = 1.
            print([model.pdf(point*i) for i in range(len(h.choices))])
        except AttributeError:
            pass
    #
    # print(config_generator.get_config(1.0))
    # print('Finished...')

