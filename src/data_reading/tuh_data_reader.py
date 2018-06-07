from src.data_reading.base_data_reader import BaseDataReader
import mne
import os
import numpy as np
from src.utils import load_dict
import logging
import random


logger = logging.getLogger(__name__)


# We assume that the data is available in a given structure:
# If you want to know how the data is prepared then
# look at the src/data_preparation/tuh_data_generator.py
# data_path
#   train
#       data
#           00000003_s02_a01_Age_70_Gender_F_raw.fif
#       info
#           00000003_s02_a01_Age_70_Gender_F.p
#           ...
#   validation
#        ...
#   test
#       ...
#
# Files inside the 'data' directory contain the data (channel normalized). One file per one recording.
# Files inside the 'info' directory contain the metadata in a json format:
#   "age": 70,  - Age of the subject
#   "anomaly": 0, - Anomaly label for the recording (1 means that there was anomaly detected inside the data)
#   "file_name": "...", - File path to the original data file used to create corresponding .fif file
#   "gender": "F", - Gender of the subject
#   "number": "00000003", - Subject specific number
#   "sequence_name": "00000003_s02_a01" - Unique id that can be used to identify the recording
# Cast all prediction problems (all label_types) as  classification problem

class AnomalyDataReader(BaseDataReader):
    label_age = 'age'
    label_age_class = 'age_class'
    label_anomaly_class = 'anomaly'
    label_gender_class = 'gender_class'
    label_types = [label_age, label_age_class, label_anomaly_class, label_gender_class]

    normalization_none = 'none'
    normalization_separate = 'separate'
    normalization_types = [normalization_none, normalization_separate]

    class ExampleInfo(BaseDataReader.BaseExampleInfo):
        def __init__(self, info_dict, label_type, offset_size=0, random_mode=0, limit_duration=None,
                     use_augmentation=0):
            super().__init__(example_id=info_dict['sequence_name'], offset_size=offset_size, random_mode=random_mode)

            self.label_type = label_type
            self.use_augmentation = use_augmentation
            self.age = (info_dict['age'] - 49.295620438) / 17.3674915241
            # Use normalized age for training, network does not have to learn bias
            if self.label_type == 'age':
                self.label = self.age
            else:
                self.label = info_dict[label_type]

            self.mean = np.array(info_dict['mean'], dtype=np.float32)
            self.std = np.array(info_dict['std'], dtype=np.float32)

            self.file_handler = mne.io.read_raw_fif(info_dict['data_file'], preload=False, verbose='error')
            self.length = self.file_handler.n_times
            self.length = self.length if limit_duration is None else min(self.length, limit_duration)

            if label_type == AnomalyDataReader.label_age:
                self.context = np.array([[info_dict['gender'] == 'M', info_dict['gender'] == 'F']]).astype(np.float32)
            elif label_type == AnomalyDataReader.label_age_class:
                self.context = np.array([[info_dict['gender'] == 'M', info_dict['gender'] == 'F']]).astype(np.float32)
            elif label_type == AnomalyDataReader.label_anomaly_class:
                self.context = np.array([[info_dict['gender'] == 'M', info_dict['gender'] == 'F',  self.age]])\
                    .astype(np.float32)
            elif label_type == AnomalyDataReader.label_gender_class:
                self.context = np.array([self.age]).astype(np.float32)
            else:
                raise NotImplementedError('Can not create context for this label_type %s' % label_type)

        # Scale from 0.5 to 2.0
        @staticmethod
        def random_scale():
            r = random.uniform(-1, 1)
            return 2**r

        def get_length(self):
            return self.length

        def read_data(self, serialized):
            index, sequence_size = serialized
            data = self.file_handler.get_data(None, index, index+sequence_size).astype(np.float32)
            data = np.transpose(data)
            # Now data has shape time x features

            # Normalize
            data = ((data - self.mean) / self.std)

            # Rescale by random value from log_uniform (0.5, 2) and revert in time with 50% chance
            if self.use_augmentation:
                data *= self.random_scale()

                revert = random.choice([0, 1])
                if revert:
                    data = data[::-1, :]

            # Time might be needed for some advanced models like PhasedLSTM
            time = np.reshape(np.arange(index, index + sequence_size), newshape=[sequence_size, 1])

            if self.label_type == 'age':
                label = np.array([[self.label]] * sequence_size)
                label = label.astype(np.float32)
            else:
                label = np.array([self.label] * sequence_size)

            return data, time, label, self.example_id, self.context

    @staticmethod
    def add_arguments(parser):
        BaseDataReader.add_arguments(parser)
        parser.add_argument("label_type", type=str, choices=AnomalyDataReader.label_types,
                            help="Path to the directory containing the data")
        parser.add_argument("normalization_type", type=str, dest='normalization_type',
                            choices=AnomalyDataReader.normalization_types,
                            help="How to normalize the data.")
        parser.add_argument("use_augmentation", type=int, choices=[0, 1], default=0,
                            help="Will add some shift and rescale the data."
                                 "Idea is to make the network invariant to those transformations")
        parser.add_argument("filter_gender", type=str, choices=['None', 'M', 'F'], default='None',
                            help="Will train using only male of female ")

        return parser

    def _initialize(self, label_type, normalization_type, use_augmentation, filter_gender, **kwargs):
        self.label_type = label_type
        self.normalization_type = normalization_type
        self.use_augmentation = use_augmentation
        self.filter_gender = filter_gender
            
        if use_augmentation and self.data_type != BaseDataReader.Train_Data:
            logger.warning('For Validation and Test we disable data augmentation')
            self.use_augmentation = 0

        if label_type == 'age' and self.balanced:
            logger.warning('Balancing not implemented for age regression. Set to 0')
            self.balanced = 0

        if self.batch_size % 2 == 1 and self.balanced:
            logger.warning('Trying to set batch size to odd number while you want to use balanced minibatch. '
                           'Increasing batch size by one from %s to %s' % (self.batch_size, self.batch_size+1))
            self.batch_size += 1

        self.limit_examples = None if self.limit_examples <= 0 else self.limit_examples
        self.limit_duration = None if self.limit_duration <= 0 else self.limit_duration

        logger.debug('Initialized %s AnomalyDataReader with parameters:' % self.data_type.title())
        logger.debug('label_type: %s' % self.label_type)
        logger.debug('limit_examples: %s' % self.limit_examples)
        logger.debug('limit_duration: %s' % self.limit_duration)

    @staticmethod
    def load_info_dicts(data_path, folder_name):
        info_dir = os.path.join(data_path, folder_name, 'info')
        info_files = sorted(os.listdir(info_dir))
        info_dicts = [load_dict(os.path.join(info_dir, i_f)) for i_f in info_files]

        for info_dict, info_file in zip(info_dicts, info_files):
            info_dict['data_file'] = os.path.join(data_path, folder_name, 'data', info_file[:-2] + '_raw.fif')
            # Compute additional fields (used for new labels and context information)
            info_dict['age_class'] = 1 if info_dict['age'] >= 49 else 0
            info_dict['gender_class'] = 1 if info_dict['gender'] == 'M' else 0

        return info_dicts

    def _create_examples(self):
        if self.normalization_type != self.normalization_none:
            if self.limit_duration is not None:
                logger.warning('Will limit example duration but will compute normalization statistics from full '
                               'recordings.')
        if self.normalization_type != self.normalization_separate:
            if self.use_augmentation:
                logger.warning('Augmentation was only designed for \"separate\" normalization.')

        # Train and Validation are located inside the 'train' folder
        if self.data_type == self.Validation_Data or self.data_type == self.Train_Data:
            folder_name = 'train'
        elif self.data_type == self.Test_Data:
            folder_name = 'test'
        else:
            raise NotImplementedError('data_type is not from the set {train, validation, test}')

        # Load data into dictionaries from the info json files
        info_dicts = self.load_info_dicts(self.data_path, folder_name)

        if self.label_type == 'gender_class':
            info_dicts = [info_dict for info_dict in info_dicts if info_dict['gender'] != 'X']

        # Filter out based on the gender if that is what user wants
        if self.filter_gender in ['M', 'F']:
            logger.warning('Will only use recordings with gender: %s' % self.filter_gender)
            info_dicts = [info_dict for info_dict in info_dicts if info_dict['gender'] == self.filter_gender]

        labels = list(set([info_dict[self.label_type] for info_dict in info_dicts]))

        # Sanity check if all labels are present in the dataset for classification (for age we do regression)
        if len(labels) != self.output_size(self.label_type) and self.label_type != 'age':
            error = 'Not all labels present in the dataset, declared %d, detected %d' % \
                    (self.output_size(self.label_type), len(labels))
            logger.error(error)
            raise RuntimeError(error)

        logger.debug('Will use %s as a label type' % self.label_type)
        logger.debug('Create info objects for the files (Number of all sequences: %s' % len(info_dicts))

        # Get corresponding cross validation data.
        info_dicts = self.cv_split(info_dicts)

        if self.normalization_type == self.normalization_none:
            logger.debug('Will not normalize the data.')
            for info_dict in info_dicts:
                info_dict['mean'] = 0.0
                info_dict['std'] = 1.0
        elif self.normalization_type == self.normalization_separate:
            logger.debug('Will normalize each recording separately.')
        else:
            raise NotImplementedError('This normalization (%s) is not implemented' % self.normalization_type)

        # Create examples
        for i, label in enumerate(labels):
            label_info_dicts = [info_dict for info_dict in info_dicts if info_dict[self.label_type] == label]
            label_info_dicts = label_info_dicts[:self.limit_examples]

            self.examples.append([AnomalyDataReader.ExampleInfo(label_info_dict, self.label_type,
                                                                self.offset_size, self.random_mode,
                                                                self.limit_duration, self.use_augmentation)
                                  for (j, label_info_dict) in enumerate(label_info_dicts)])

    @staticmethod
    # Has to be a static method, context_size is required when creating the model,
    # DataReader can't be instantiated properly before the model is created
    def context_size(label_type, **kwargs):
        if label_type == 'gender_class':
            return 1
        elif label_type in ['age_class', 'age']:
            return 2
        elif label_type == 'anomaly':
            return 3

    @staticmethod
    def input_size(**kwargs):
        return 21

    @staticmethod
    def output_size(label_type, **kwargs):
        # For age we do regression
        if label_type == 'age':
            return 1
        return 2

