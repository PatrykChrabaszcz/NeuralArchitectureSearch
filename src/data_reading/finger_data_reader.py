from src.data_reading.base_data_reader import BaseDataReader
import logging
import numpy as np
from scipy.io import loadmat
import os
from multiprocessing import sharedctypes
import random

logger = logging.getLogger(__name__)


class FingerDataReader(BaseDataReader):
    class ExampleInfo(BaseDataReader.BaseExampleInfo):
        def __init__(self, example_id, data_ctypes, data_shape, labels_ctypes, labels_shape,
                     offset_size=0, random_mode=0, limit_duration=None):
            super().__init__(example_id=example_id, offset_size=offset_size, random_mode=random_mode)

            self.data = np.frombuffer(data_ctypes, dtype=np.float32, count=int(np.prod(data_shape)))
            self.data.shape = data_shape
            self.labels = np.frombuffer(labels_ctypes, dtype=np.float32, count=int(np.prod(labels_shape)))
            self.labels.shape = labels_shape

            self.length = self.data.shape[0]
            self.length = self.length if limit_duration is None else min(self.length, limit_duration)

            self.context = None

        def get_length(self):
            return self.length

        def read_data(self, serialized):
            index, sequence_size = serialized

            data = self.data[index: index+sequence_size]
            # Time might be needed for some advanced models like PhasedLSTM
            time = np.reshape(np.arange(index, index + sequence_size), newshape=[sequence_size, 1])

            label = self.labels[index: index + sequence_size]

            assert not np.isnan(data).any(), 'Data has nans %s' % data
            assert not np.isnan(label).any(), 'Label has nans %s' % label

            return data, time, label, self.example_id, self.context

        # We need to overwrite this method, all examples read from the same array, we need to read from a
        # completely different points within the sequence
        def reset(self, sequence_size, randomize=False):
            super().reset(sequence_size)

            # Additional behaviour (overwrite the self.curr_index with random point in the file (not just a phase)
            if randomize and not self.done:
                margin = self.get_length() - self.sequence_size
                self.curr_index = random.randint(0, margin)

    @staticmethod
    def add_arguments(parser):
        BaseDataReader.add_arguments(parser)
        parser.add_argument("subject", type=int, choices=[1, 2, 3], default=1,
                            help="Number of the patient.")
        parser.add_argument('fingers', default='0,1,2,4', type=str,
                            help="Which fingers will be used, comma separated without spaces")
        return parser

    def _initialize(self, subject, fingers, **kwargs):
        self.subject = subject
        self.fingers = [int(f) for f in fingers.split(',')]

        if self.balanced:
            logger.warning('Class balancing is not implemented for this dataset.')

    def _create_examples(self):
        logger.info('Read information about all sample files from the dataset')

        # Extract file for the subject
        files = [os.path.join(self.data_path, 'sub%d_comp.mat' % subject) for subject in range(1, 4)]
        file = files[self.subject]

        # Load the data. Matrix shape [Time x Features]
        data_dict = loadmat(file)
        data = data_dict['train_data'].astype(np.float32)

        data -= data.mean()
        data /= data.std()
        # Filter out fingers
        labels = data_dict['train_dg'].astype(np.float32)[:, self.fingers]

        # Get the data from cross-validation split
        # Problematic dataset, not sure yet how to connect training data if validation split is in the middle
        if self.data_type == self.Validation_Data or self.data_type == self.Train_Data:
            start = int(self.cv_k / self.cv_n * data.shape[0])
            end = int((self.cv_k + 1) / self.cv_n * data.shape[0])

            logger.info("Using CV split cv_n: %s, cv_k: %s, start: %s, end: %s" % (self.cv_n, self.cv_k, start, end))

            if self.data_type == self.Train_Data:
                data = np.ascontiguousarray(np.concatenate([data[:start, :], data[end:, :]], axis=0))
                labels = np.ascontiguousarray(np.concatenate([labels[:start, :], labels[end:, :]], axis=0))
            else:
                data = np.ascontiguousarray(data[start:end, :])
                labels = np.ascontiguousarray(labels[start:end, :])
        else:
            raise RuntimeError('Test split is not implemented right now for this dataset (Labels are not provided)')

        # Make a shared array, read only access so no need to lock
        data_shape = data.shape
        labels_shape = labels.shape

        data.shape = data.size
        labels.shape = labels.size

        self.data_ctypes = sharedctypes.RawArray('f', data)
        self.labels_ctypes = sharedctypes.RawArray('f', labels)

        logger.info('Create info objects for the files')

        # For validation and test we only have one example
        # For training we create multiple examples such that different parts of the file will be used in one
        # mini-batch
        number_of_examples = 128 if self.data_type == BaseDataReader.Train_Data else 1
        self.examples.append([FingerDataReader.ExampleInfo(i, self.data_ctypes, data_shape, self.labels_ctypes,
                                                           labels_shape, offset_size=self.offset_size,
                                                           random_mode=self.random_mode)
                              for i in range(number_of_examples)])

    @staticmethod
    def context_size(**kwargs):
        # What kind of context can be used in this dataset ?
        # We could for example encode patient code and train one model for multiple patients instead of separate
        # model for each patient
        return 0

    @staticmethod
    def input_size(**kwargs):
        return 48

    @staticmethod
    def output_size(fingers, **kwargs):
        return len(fingers.split(','))
