from src.hpbandster.results_logger import ResultLogger
from src.experiment_arguments import ExperimentArguments
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

stdv = 17.3674915241
sns.set_style("whitegrid")


class Visualizer:
    def __init__(self, working_dir, config_space_file):
        self.working_dir = working_dir
        self.config_space_file = config_space_file
        self.result_logger = ResultLogger(working_dir=working_dir)
        self.config_space = ExperimentArguments.read_configuration_space(config_space_file)

        self.results = {}
        experiment_start_time = np.inf

        self.alpha = 0.4
        self.size = 40
        colors = ["windows blue", "amber", "green", "red", ]
        self.pal = sns.xkcd_palette(colors)

        self.lines = {}

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
                result = 100 + float(result['loss']) * 100
                #result = np.sqrt(float(result['loss']) * stdv**2)
            t_finished = int(result_timestamps['finished'])
            point = ((t_finished - experiment_start_time) / 3600, result, info['model_based_pick'])
            if point[0] > 170:
               continue
            try:
                self.results[budget].append(point)
            except KeyError:
                self.results[budget] = [point]

            try:
                self.lines[configuration].append(point)
            except KeyError:
                self.lines[configuration] = [point]

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
        for c, (key, values) in zip(self.pal, sorted(self.results.items())):
            print(c)
            mb_values = [v[:2] for v in values if v[2] is True]
            mf_values = [v[:2] for v in values if v[2] is False]

            try:
                print('A')
                x, y = [list(a) for a in zip(*mb_values)]

                plt.scatter(x, y, c=c, s=self.size, linewidths=2, edgecolors='black', alpha=self.alpha,
                            label=str('Model Based %s' % int(key)))
                #sns.regplot(np.array(x), np.array(y), color=c, ci=99)
            except Exception as e:
                print(e)
                pass

            try:
                x, y = [list(a) for a in zip(*mf_values)]
                plt.scatter(x, y, c=c, s=self.size, alpha=self.alpha, label=str('Model Free %s' % int(key)))
            except Exception as e:
                print(e)
                pass
            plt.xlim(-1, 80)
            plt.ylim(-1, 80)
            plt.xlabel("Time [hours]", fontsize=23)
            plt.ylabel("Classification Error [%]", fontsize=23) #
            #plt.ylabel("RMSE", fontsize=25)
            plt.tick_params(axis='both', which='major', labelsize=22)
            plt.tick_params(axis='both', which='minor', labelsize=21)

    def plot_best_points(self):
        for c, (key, values) in zip(self.pal, sorted(self.best_points.items())):
            x, y = [list(a) for a in zip(*values)]
            plt.step(x, y, c=c, linewidth=3, where='post', alpha=0.8)

            print(y)

    def plot_lines(self):
        for key, value in self.lines.items():
            if len(value) > 1:
                x = [v[0] for v in value]
                y = [v[1] for v in value]

                plt.plot(x, y, c='gray', alpha=0.2)


if __name__ == '__main__':
    mode = 'LuFi'

    if mode in ['LuFi', 'GuJo', 'MaJa', 'RoSc']:

    working_dir = '/mhome/chrabasp/EEG_Results/BBCI_BO/LuFiMoSc'
    working_dir = '/mhome/chrabasp/EEG_Results/BBCI_BO/GuJoMoSc'
    working_dir = '/mhome/chrabasp/EEG_Results/BBCI_BO/MaJaMoSc'
    working_dir = '/mhome/chrabasp/EEG_Results/BBCI_BO/RoScMoSc'
    config_space_file = '/mhome/chrabasp/Workspace/EEG/config/bbci/bbci_cnn_rnn.pcs'


    #working_dir = '/mhome/chrabasp/EEG_Results/MNIST/Random'
    #working_dir = '/mhome/chrabasp/EEG_Results/MNIST/BO'
    #config_space_file = '/mhome/chrabasp/Workspace/EEG/config/mnist_simple.pcs'


    #working_dir = '/mhome/chrabasp/EEG_Results/BBCI_BO/LuFiMoSc3S001'
    #working_dir = '/mhome/chrabasp/EEG_Results/BO_Anomaly_6'
    #working_dir = '/mhome/chrabasp/EEG_Results/TCN_Gender_BO'
    #working_dir = '/mhome/chrabasp/EEG_Results/BO_Anomaly_Gender_60'
    #working_dir = '/mhome/chrabasp/EEG_Results/BO_Anomaly_Age_60_fixed'
    #working_dir = '/mhome/chrabasp/EEG_Results/TCN_Age_BO'
    #working_dir = '/mhome/chrabasp/EEG_Results/TCN_Anomaly_BO'
    #config_space_file = '/mhome/chrabasp/Workspace/EEG/config/bbci/bbci_cnn_rnn.pcs'
    #config_space_file = '/mhome/chrabasp/Workspace/EEG/config/anomaly_dataset/anomaly_tcn.pcs'
    #config_space_file = '/mhome/chrabasp/Workspace/EEG/config/anomaly_dataset/anomaly_simple.pcs'

    plt.figure(figsize=(6, 7))
    visualizer = Visualizer(working_dir=working_dir, config_space_file=config_space_file)
    visualizer.plot_lines()
    visualizer.plot()
    visualizer.plot_best_points()
    plt.tight_layout(pad=1.12, rect=[0, 0.03, 1, 0.95])
    #plt.tight_layout(pad=1.12, rect=[0, 0.2, 1.0, 0.8])

    #plt.legend(fontsize=14, ncol=4, fancybox=True, shadow=True, frameon=True, loc='upper center', markerscale=3)
    plt.title('MaJa', fontsize=25)
    plt.show()
