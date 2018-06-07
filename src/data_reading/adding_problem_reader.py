from src.data_reading.base_data_reader import BaseDataReader
import numpy as np
import logging


logger = logging.getLogger(__name__)


class AddingProblemReader(BaseDataReader):
    class ExampleInfo(BaseDataReader.BaseExampleInfo):
        def __init__(self, example_id, length):
            # We set offset_size and random_mode to 0 because we will not use truncated BPTT for this reader
            super().__init__(example_id=example_id, offset_size=0, random_mode=0)
            self.length = length
            self.context = None

        def get_length(self):
            return self.length

        def read_data(self, serialized):
            index, sequence_size = serialized
            assert index == 0, 'For Adding Problem dataset only reading from the beginning is considered.'
            assert sequence_size == self.length, 'Values for length (%s) and sequence size (%s) do not match' % \
                                                 (self.length, sequence_size)
            # Dimension, time x 2
            random_values = np.random.uniform(0, 1, size=sequence_size).astype(np.float32)
            add_indices = np.zeros_like(random_values, dtype=np.float32)
            first_num = np.random.randint(0, sequence_size//2)
            second_num = np.random.randint(sequence_size//2, sequence_size)
            add_indices[first_num] = 1
            add_indices[second_num] = 1

            data = np.stack([random_values, add_indices], axis=1)
            # Time might be needed for some advanced models like PhasedLSTM
            time = np.reshape(np.arange(0, sequence_size), newshape=[sequence_size, 1])

            label = np.array([[random_values[first_num] + random_values[second_num]]] * sequence_size)

            return data, time, label, self.example_id, self.context

    def _initialize(self, **kwargs):
        pass

    def _create_examples(self):
        # Lets say that one epoch for any data_type will consist of 10 * batch_size examples
        # Data is generated on the fly
        num_examples = self.batch_size * 10

        logger.info('Creating %s examples' % num_examples)
        examples = [self.ExampleInfo(example_id=i, length=self.sequence_size) for i in range(num_examples)]
        self.examples.append(examples)

    @staticmethod
    # Has to be a static method, context_size is required when creating the model,
    # DataReader can't be instantiated properly before the model is created
    def context_size(**kwargs):
        return 0

    @staticmethod
    def input_size(**kwargs):
        return 2

    @staticmethod
    def output_size(**kwargs):
        return 1
