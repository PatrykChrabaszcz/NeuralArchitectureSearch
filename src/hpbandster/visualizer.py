from src.hpbandster.results_logger import ResultLogger
from src.experiment_arguments import ExperimentArguments
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Visualizer:
    def __init__(self, working_dir, config_space_file):
        self.working_dir = working_dir
        self.config_space_file = config_space_file
        self.result_logger = ResultLogger(working_dir=working_dir)
        self.config_space = ExperimentArguments.read_configuration_space(config_space_file)

        self.results = {}
        experiment_start_time = np.inf

        self.alpha = 0.8
        self.size = 50

        # First find starting time
        for configuration, result in self.result_logger.get_results(self.config_space):
            budget, result_timestamps, result, exception = result
            t_started = int(result_timestamps['started'])
            experiment_start_time = min(t_started, experiment_start_time)

        for configuration, result, info in self.result_logger.get_results(self.config_space, with_info=True):
            budget, result_timestamps, result, exception = result
            if result is None:
                result = np.inf
            else:
                result = float(result['loss'])

            t_finished = int(result_timestamps['finished'])
            point = ((t_finished - experiment_start_time) / 3600, result, info['model_based_pick'])
            try:
                self.results[budget].append(point)
            except KeyError:
                self.results[budget] = [point]

        self.best_points = {}
        for key, values in self.results.items():
            self.best_points[key] = []
            values = sorted(values, key=lambda c: c[0])
            best = np.inf
            for time, result, mb in values:
                if result < best:
                    best = result
                    self.best_points[key].append((time, result))

    def plot(self):
        pal = sns.color_palette(n_colors=len(self.results.keys()))
        for c, (key, values) in zip(pal, sorted(self.results.items())):
            mb_values = [v[:2] for v in values if v[2] is True]
            mf_values = [v[:2] for v in values if v[2] is False]

            x, y = [list(a) for a in zip(*mb_values)]
            plt.scatter(x, y, c=c, s=self.size, linewidths=2, edgecolors='black', alpha=self.alpha,
                        label=str('Model Based %s' % key))

            x, y = [list(a) for a in zip(*mf_values)]
            plt.scatter(x, y, c=c, s=self.size, alpha=self.alpha, label=str('Model Free %s' % key))

    def plot_best_points(self):
        pal = sns.color_palette(n_colors=len(self.best_points.keys()))
        for c, (key, values) in zip(pal, sorted(self.best_points.items())):
            x, y = [list(a) for a in zip(*values)]
            plt.step(x, y, c=c, linewidth=3)


if __name__ == '__main__':

    working_dir = '/mhome/chrabasp/EEG_Results/BO_Anomaly_Age_60'
    config_space_file = '/mhome/chrabasp/Workspace/EEG/config/anomaly_dataset/anomaly_simple.pcs'

    visualizer = Visualizer(working_dir=working_dir, config_space_file=config_space_file)
    visualizer.plot()
    visualizer.plot_best_points()

    plt.legend()
    plt.show()
