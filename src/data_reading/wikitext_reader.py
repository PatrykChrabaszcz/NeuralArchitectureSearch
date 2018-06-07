from src.data_reading.base_data_reader import BaseDataReader
import os
import numpy as np
import logging
import random
import json
from multiprocessing import sharedctypes

logger = logging.getLogger(__name__)


# This does not use multi-threading at all
class WikiTextReader(BaseDataReader):
    class ExampleInfo(BaseDataReader.BaseExampleInfo):
        def __init__(self, example_id, data_ctypes, offset_size, random_mode,):
            super().__init__(example_id=example_id, offset_size=offset_size, random_mode=random_mode)
            # This assert is also fulfilled when limit_duration is None
            self.data = np.frombuffer(data_ctypes, dtype=np.float32)
            self.length = len(self.data)

            # Nothing that comes to my mind and can be used as context for this dataset
            self.context = None

        def get_length(self):
            return self.length

        def read_data(self, serialized):
            index, sequence_size = serialized

            data = self.data[index: index + sequence_size]
            # Time might be needed for some advanced models like PhasedLSTM
            time = np.reshape(np.arange(index, index + sequence_size), newshape=[sequence_size, 1])

            label = self.data[index+1: index + sequence_size + 1]

            # In case where we pick the data from the end and miss one label just repeat last word
            if len(label) != len(data):
                label = np.concatenate((label, label[-1:]))

            return data.astype(np.int64), time, label.astype(np.int64), self.example_id, self.context

        # We need to overwrite this method, all examples read from the same array, we need to read from a
        # completely different points within the sequence
        def reset(self, sequence_size, randomize=False):
            super().reset(sequence_size)

            # Additional behaviour (overwrite the self.curr_index with random point in the file (not just a phase)
            if randomize and not self.done:
                margin = self.get_length() - self.sequence_size
                self.curr_index = random.randint(0, margin)

    def _initialize(self, **kwargs):
        pass

    def _create_examples(self):
        # Creates data if not yet created
        corpus = Corpus(self.data_path)

        name = 'valid' if self.data_type == self.Validation_Data else self.data_type
        data = getattr(corpus, name)

        # Make a shared array, read only access so no need to lock
        self.data_ctypes = sharedctypes.RawArray('f', data)

        logger.info('Create info objects for the files')

        # For validation and test we only have one example
        # For training we create multiple examples such that different parts of the file will be used in one
        # mini-batch
        number_of_examples = 128 if self.data_type == BaseDataReader.Train_Data else 1
        self.examples.append([self.ExampleInfo(example_id=i, data_ctypes=self.data_ctypes,
                                               offset_size=self.offset_size, random_mode=self.random_mode)
                              for i in range(number_of_examples)])

        # CV Not implemented for now for this dataset
        logger.warning('For WikiText dataset separate validation set is provided, '
                       'currently cv_n and cv_k settings have no effect!')

    @staticmethod
    # Has to be a static method, context_size is required when creating the model,
    # DataReader can't be instantiated properly before the model is created
    def context_size(**kwargs):
        return 0

    @staticmethod
    def input_size(data_path, **kwargs):
        # Here it will be equal to the number of tokens
        # We will cache the number of tokens after the first usage

        try:
            dictionary = Dictionary()
            dictionary.load(data_path)
        except FileNotFoundError:
            corpus = Corpus(data_path)
            dictionary = corpus.dictionary

        return len(dictionary)

    @staticmethod
    def output_size(**kwargs):
        # Generative model so input_size == output_size
        return WikiTextReader.input_size(**kwargs)


# https://github.com/pytorch/examples/tree/master/word_language_model
# Little bit adjusted for our own purposes
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def save(self, path):
        file_path = os.path.join(path, 'dictionary.json')
        info = {'word2idx': self.word2idx,
                'idx2word': self.idx2word
                }

        with open(file_path, 'w') as f:
            json.dump(info, f)

    def load(self, path):
        file_path = os.path.join(path, 'dictionary.json')
        with open(file_path, 'r') as f:
            info = json.load(f)
        self.word2idx = info['word2idx']
        self.idx2word = info['idx2word']

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        try:
            self.train = np.load(os.path.join(path, 'train.npy'))
            self.valid = np.load(os.path.join(path, 'valid.npy'))
            self.test = np.load(os.path.join(path, 'test.npy'))
            self.dictionary.load(os.path.join(path))

        except FileNotFoundError:
            self.train = self.tokenize(os.path.join(path, 'train.txt'))
            self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
            self.test = self.tokenize(os.path.join(path, 'test.txt'))

            # Cache tokenized data
            np.save(os.path.join(path, 'train.npy'), self.train)
            np.save(os.path.join(path, 'valid.npy'), self.valid)
            np.save(os.path.join(path, 'test.npy'), self.test)
            self.dictionary.save(path)

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = np.zeros(shape=tokens, dtype=np.int64)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


if __name__ == '__main__':
    corpus = Corpus('/mhome/chrabasp/data/WikiText')
    print(len(corpus.dictionary))
    print(corpus.test.size)
    print(corpus.test.shape)

