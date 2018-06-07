from braindecode.datautil.signalproc import exponential_running_standardize
import resampy
import numpy as np
import logging
from src.utils import save_dict
import glob
import re
import os
import mne
import click
import json


log = logging.getLogger()


class DataGenerator:
    """
    Class used to generate preprocessed abnormal/normal recordings.
        - Subsamples recordings to desired frequency
        -
        Original raw recordings will be sub-sampled to have
    the same frequency specified by the user (default 100). Specified number of seconds will be removed both from
    the beginning and from the end of each recording. Reach output recording will be also limited to specified
    duration.
    """
    wanted_electrodes = {
        'EEG': ['EEG A1-REF', 'EEG A2-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG CZ-REF', 'EEG F3-REF', 'EEG F4-REF',
                'EEG F7-REF', 'EEG F8-REF', 'EEG FP1-REF', 'EEG FP2-REF', 'EEG FZ-REF', 'EEG O1-REF', 'EEG O2-REF',
                'EEG P3-REF', 'EEG P4-REF', 'EEG PZ-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF'],
        'EKG1': ['EEG EKG1-REF'],
        'EKG': ['ECG EKG-REF']

    }

    # More like a namespace than class
    class Key:
        @staticmethod
        def session_key(file_name):
            return re.findall(r'(s\d{2})', file_name)

        @staticmethod
        def natural_key(file_name):
            key = [int(token) if token.isdigit() else None for token in re.split(r'(\d+)', file_name)]
            return key

        @staticmethod
        def time_key(file_name):
            splits = file_name.split('/')
            [date] = re.findall(r'(\d{4}_\d{2}_\d{2})', splits[-2])
            date_id = [int(token) for token in date.split('_')]
            recording_id = DataGenerator.Key.natural_key(splits[-1])
            session_id = DataGenerator.Key.session_key(splits[-2])
            return date_id + session_id + recording_id

    class FileInfo:
        # Can throw a ValueError exception if file is wrongly formatted
        def __init__(self, file_path, preload=False):
            """ read info from the edf file without loading the data. loading data is done in multiprocessing since it takes
            some time. getting info is done before because some files had corrupted headers or weird sampling frequencies
            that caused the multiprocessing workers to crash. therefore get and check e.g. sampling frequency and duration
            beforehand
            :param file_path: path of the recording file
            :return: file name, sampling frequency, number of samples, number of signals, signal names, duration of the rec
            """
            self.edf_file = None
            self.sampling_frequency = None
            self.n_samples = None
            self.n_signals = None
            self.signal_names = None
            self.duration = None

            edf_file = mne.io.read_raw_edf(file_path, preload=preload, verbose='error')

            # Some recordings have a very weird sampling frequency. Check twice before skipping the file
            sampling_frequency = int(edf_file.info['sfreq'])
            if sampling_frequency < 10:
                sampling_frequency = 1 / (edf_file.times[1] - edf_file.times[0])
                if sampling_frequency < 10:
                    self.sampling_frequency = sampling_frequency
                    raise RuntimeError('Weird sampling frequency (%g)for the file %s' % (sampling_frequency, file_path))

            self.edf_file = edf_file
            self.sampling_frequency = sampling_frequency
            self.n_samples = edf_file.n_times
            self.signal_names = edf_file.ch_names
            self.n_signals = len(self.signal_names)
            # Some weird sampling frequencies are at 1 hz or below, which results in division by zero
            self.duration = self.n_samples / max(sampling_frequency, 1)

            # For some strange reason beyond my knowledge MNE library does not extract Age and Gender
            with open(file_path, 'rb') as f:
                header = f.read(88)
                patient_id = header[8:88].decode('ascii')
                [age] = re.findall("Age:(\d+)", patient_id)
                [gender] = re.findall("\s([F|M|X])\s", patient_id)
                [number] = re.findall("^(\d{8})", patient_id)
                [date] = re.findall("s\d{2}_(\d{4}_\d{2}_\d{2})", file_path)

            self.age = age
            self.gender = gender
            self.number = number
            self.sequence_name = os.path.basename(file_path)[:-4]
            self.recording_date = date

            # TODO: We would like to read some more information from the description files
            # txt_file_path = file_path[:-8] + '.txt'

    def __init__(self, data_path, cache_path, secs_to_cut, sampling_freq, duration_mins, max_abs_value, use_ekg,
                 exponential_normalization, subsample_filter='kaiser_fast', version='v1.1.2'):
        """
        :param data_path:
            Path to the original data
        :param cache_path:
            Path to the derived data (User has to make sure that this directory is empty when new options are used)
        :param version:
            Dataset version
        """
        self.data_path = data_path
        self.cache_path = cache_path
        self.use_ekg = use_ekg
        self.subsample_filter = subsample_filter
        self.version = version

        self.secs_to_cut_at_start_end = secs_to_cut
        self.sampling_freq = sampling_freq
        self.duration_mins = duration_mins
        self.max_abs_value = max_abs_value
        self.exponential_normalization = exponential_normalization

        os.makedirs(self.cache_path)
        with open(os.path.join(self.cache_path, 'data_generator_info.json'), 'w') as f:
            json.dump(vars(self), f)

    @staticmethod
    def read_all_file_names(path, extension='.edf', key=Key.time_key):
        file_paths = glob.glob(os.path.join(path, '**/*' + extension), recursive=True)
        return sorted(file_paths, key=key)

    # Return File names for ["train", "eval"] X ["normal", "abnormal"]
    def _file_names(self, train=True, normal=True):
        mode = 'train' if train else 'eval'
        label = 'normal' if normal else 'abnormal'
        sub_path = '{label}{version}/{version}/' \
                   'edf/{mode}/{label}/'.format(mode=mode, label=label, version=self.version)
        path = os.path.join(self.data_path, sub_path)
        return self.read_all_file_names(path, key=DataGenerator.Key.time_key)

    # Get all file names for train and test data, sorted according to the time key
    def _get_all_sorted_file_names_and_labels(self, train=True):
        normal_file_names = self._file_names(train=train, normal=True)
        abnormal_file_names = self._file_names(train=train, normal=False)

        all_file_names = normal_file_names + abnormal_file_names
        all_file_names = sorted(all_file_names, key=DataGenerator.Key.time_key)

        abnormal_counts = [file_name.count('abnormal') for file_name in all_file_names]
        assert set(abnormal_counts) == {1, 3}
        labels = np.array(abnormal_counts) == 3
        labels = labels.astype(np.int64)

        return all_file_names, labels

    def _load_file(self, file_name, preprocessing_functions, sensor_types=('EEG',)):
        wanted_electrodes = []
        for sensor_type in sensor_types:
            wanted_electrodes.extend(DataGenerator.wanted_electrodes[sensor_type])

        # This guy can throw an exception
        file_info = DataGenerator.FileInfo(file_name, preload=True)

        log.info("Load file %s" % file_name)
        cnt = file_info.edf_file.pick_channels(wanted_electrodes)

        if not np.array_equal(cnt.ch_names, wanted_electrodes):
            raise RuntimeError('Not all channels available')

        # From volt to microvolt
        data = (cnt.get_data() * 1e6).astype(np.float32)
        fs = cnt.info['sfreq']

        if preprocessing_functions is not None:
            for preprocessing_function in preprocessing_functions:
                log.info(preprocessing_function)

                data, fs = preprocessing_function(data, fs)
                if data.dtype == np.float64:
                    data = data.astype(np.float32)
                assert (data.dtype == np.float32) and (type(fs) == float), (data.dtype, type(fs))

        # Extract some additional info: Age, Gender, number
        anomaly_cnt = file_name.count('abnormal')
        assert(anomaly_cnt in [1, 3])
        info_dictionary = {
            'age': int(file_info.age),
            'gender': file_info.gender,
            'number': file_info.number,
            'anomaly': 1 if anomaly_cnt == 3 else 0,
            'file_name': file_name,
            'sequence_name': file_info.sequence_name,
            'recording_date': file_info.recording_date
        }

        return data, info_dictionary

    def default_preprocessing_functions(self):
        preprocessing_functions = []

        if self.secs_to_cut_at_start_end > 0:
            preprocessing_functions.append(lambda data, fs: (data[:, int(self.secs_to_cut_at_start_end * fs):-int(
                self.secs_to_cut_at_start_end * fs)], fs))

        if self.duration_mins > 0:
            preprocessing_functions.append(lambda data, fs: (data[:, :int(self.duration_mins * 60 * fs)], fs))

        preprocessing_functions.append(lambda data, fs:
                                       (resampy.resample(data, sr_orig=fs, sr_new=self.sampling_freq, axis=1,
                                                         filter=self.subsample_filter), self.sampling_freq))
        if self.max_abs_value > 0:
            preprocessing_functions.append(lambda data, fs: (
                np.clip(data, -self.max_abs_value, self.max_abs_value), fs))

        if self.exponential_normalization:
            preprocessing_functions.append(lambda data, fs: (
                                           exponential_running_standardize(data.T, init_block_size=1000,
                                                                           factor_new=0.001, eps=1e-4).T, fs))

        return preprocessing_functions

    def prepare(self):
        train_files, train_labels = self._get_all_sorted_file_names_and_labels(train=True)
        assert len(train_files) == len(train_labels) and len(train_files) != 0

        test_files, test_labels = self._get_all_sorted_file_names_and_labels(train=False)
        assert len(test_files) == len(test_labels) and len(test_files) != 0

        # Find out normalization statistics:
        preprocessing_functions = self.default_preprocessing_functions()

        ch_names = DataGenerator.wanted_electrodes['EEG']
        if self.use_ekg:
            ch_names = ch_names + DataGenerator.wanted_electrodes['EKG']

        for split_type, split_files in zip(['train', 'test'],
                                           [train_files, test_files]):

            output_data_dir = os.path.join(self.cache_path, split_type, 'data')
            output_info_dir = os.path.join(self.cache_path, split_type, 'info')
            os.makedirs(output_data_dir, exist_ok=True)
            os.makedirs(output_info_dir, exist_ok=True)

            # Could be parallelized in the future
            for i, file in enumerate(split_files):
                try:
                    sensor_types = ('EEG', 'EKG1') if self.use_ekg else ('EEG',)
                    data, info_dict = self._load_file(file, preprocessing_functions, sensor_types)
                except RuntimeError:
                    sensor_types = ('EEG', 'EKG') if self.use_ekg else ('EEG',)
                    data, info_dict = self._load_file(file, preprocessing_functions, sensor_types)

                # Find normalization for the data
                mean = np.mean(data, dtype=np.float32)
                std = np.std(data, dtype=np.float32)
                info_dict['mean'] = float(mean)
                info_dict['std'] = float(std)

                name = '%s_%s_Age_%s_Gender_%s' % (str(info_dict['recording_date']), str(info_dict['sequence_name']),
                                                   str(info_dict['age']), info_dict['gender'])
                output_file_path = os.path.join(output_data_dir, name + '_raw.fif')
                output_info_path = os.path.join(output_info_dir, name + '.p')

                info = mne.create_info(ch_names, sfreq=self.sampling_freq)
                fif_array = mne.io.RawArray(data, info)
                fif_array.save(output_file_path)

                save_dict(info_dict, output_info_path)

                print('Split Type: %s, Progress: %g' % (split_type, (i+1)/len(split_files)))


@click.command()
@click.option('--data_path', type=click.Path(exists=True), required=True)
@click.option('--cache_path', type=click.Path(), required=True)
@click.option('--secs_to_cut', default=60, help='How many seconds are removed from the beginning/end of the recording.')
@click.option('--sampling_freq', default=100.0, help='Signal will be preprocessed to the desired frequency.')
@click.option('--duration_mins', default=0, help='Duration of the recording.')
@click.option('--max_abs_value', default=800, help='Clip if channel value is grater than max_abs_value')
@click.option('--use_ekg', default=0, type=int, help='Clip if channel value is grater than max_abs_value')
@click.option('--subsample_filter', default='kaiser_fast', type=click.Choice(['kaiser_fast', 'kaiser_best']),
              help='Use kaiser_best for high quality or kaiser_fast for fast computation.')
@click.option('--exponential_normalization', default=0, type=int,
              help='If set to 1 then will perform exponential normalization using braindecode toolkit')
def main(data_path, cache_path, secs_to_cut, sampling_freq, duration_mins, max_abs_value, use_ekg, subsample_filter,
         exponential_normalization):
    print('Settings:')
    print('Data path: %s' % data_path)
    print('Cache path: %s' % cache_path)
    print('Seconds to cut (beginning and end): %d' % secs_to_cut)
    print('Sampling frequency: %d' % sampling_freq)
    print('Duration of the recording: %d' % duration_mins)
    print('Maximum absolute value: %g' % max_abs_value)
    print('Use EKG set to: %d' % use_ekg)
    print('Subsample_filter set to: %s' % subsample_filter)
    print('Exponential normalization set to: %s' % exponential_normalization)

    data_generator = DataGenerator(data_path=data_path, cache_path=cache_path, secs_to_cut=secs_to_cut,
                                   sampling_freq=sampling_freq, duration_mins=duration_mins, max_abs_value=max_abs_value,
                                   use_ekg=use_ekg, subsample_filter=subsample_filter,
                                   exponential_normalization=exponential_normalization)
    data_generator.prepare()


if __name__ == "__main__":
    main()
