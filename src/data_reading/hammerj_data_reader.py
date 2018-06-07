from src.data_reading.base_data_reader import BaseDataReader
import logging
import numpy as np
import h5py

logger = logging.getLogger(__name__)


class HammerjDataReader(BaseDataReader):
    class ExampleInfo(BaseDataReader.BaseExampleInfo):
        def __init__(self, example_id, data, labels, offset_size=0, random_mode=0):
            super().__init__(example_id=example_id, offset_size=offset_size, random_mode=random_mode)

            self.data = np.transpose(data)
            self.label = labels

            self.length = self.data.shape[0]

            self.context = None

        def get_length(self):
            return self.length

        def read_data(self, serialized):
            index, sequence_size = serialized

            # Time might be needed for some advanced models like PhasedLSTM
            time = np.reshape(np.arange(index, index + sequence_size), newshape=[sequence_size, 1])
            data = self.data[index: index+sequence_size]
            label = np.expand_dims(self.label[index: index+sequence_size], axis=1)

            return data, time, label, self.example_id, self.context

    @staticmethod
    def add_arguments(parser):
        BaseDataReader.add_arguments(parser)
        return parser

    def _initialize(self, **kwargs):
        if self.balanced:
            logger.warning('Class balancing is not implemented for this dataset.')

    def _create_examples(self):
        logger.info('Create examples for %s ' % self.data_path)

        data_list = []
        with h5py.File(self.data_path, 'r') as hf:
            trials = [hf[obj_ref] for obj_ref in hf['D'][0]]
            for idx, trial in enumerate(trials, 0):
                if idx < 20:
                    X = trial['ieeg'][:].astype(np.float32)
                    y = trial['traj'][:][:].squeeze().astype(np.float32)

                    data_list.append((X, y))

        data_list = self.cv_split(data_list)

        logger.info('Create info objects for the files')

        # For validation and test we only have one example
        # For training we create multiple examples such that different parts of the file will be used in one
        # mini-batch
        class_examples = []
        for i, (data, labels) in enumerate(data_list):
            class_examples.append(HammerjDataReader.ExampleInfo(example_id=i, data=data, labels=labels,
                                                                offset_size=self.offset_size, random_mode=0))
        self.examples.append(class_examples)\

    @staticmethod
    def context_size(**kwargs):
        # What kind of context can be used in this dataset ?
        # We could for example encode patient code and train one model for multiple patients instead of separate
        # model for each patient
        return 0

    @staticmethod
    def input_size(**kwargs):
        return 125

    @staticmethod
    def output_size(**kwargs):
        return 1
