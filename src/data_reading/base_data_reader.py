import numpy as np
import multiprocessing
import random
import logging
import signal
from queue import Empty
from src.utils import nested_list_len

# Definitions
# EXAMPLE: we use the term 'example' to define independent sequence, e.g. full sequence recording from one session.
# SEQUENCE: example can be divided into separate (dependent) sequences, e.g. example with 1000 samples can be divided
# into 10 sequences with size 100.
#
# We batch sequences from different (random) examples and use them for training.
# If we have an example of size 300(t: 0-299) with sequences of size 100 : A (t: 0-99), B(t: 100-199), C(199: 299)
# then we take care of forwarding RNN state from the end of sequence A to the beginning if sequence B.
# Because of that it is impossible to put 2 sequences from the same example into one mini-batch.


logger = logging.getLogger(__name__)


# We need to manage state
class BaseDataReader:
    class EpochDone(Exception):
        pass

    Train_Data = 'train'
    Validation_Data = 'validation'
    Test_Data = 'test'
    DataTypes = [Train_Data, Validation_Data, Test_Data]

    # This class is used to hold information about examples
    class BaseExampleInfo:
        def __init__(self,
                     example_id,
                     random_mode,
                     offset_size):
            # example_id is used to identify current RNN state for specific example.
            # We want to have an ability to process batches with random samples, but we still need to
            # use proper initial RNN state in each mini-batch iteration
            self.example_id = example_id
            # offset_size is useful when CNNs are in the first couple of layers, if CNN decreases time resolution
            # then offset_size makes sure that hidden states are matched
            self.offset_size = offset_size
            self.sequence_size = None
            self.random_mode = random_mode

            # Has to be initialized at the beginning of every epoch
            self.state = None

            self.curr_index = 0
            self.done = False
            self.blocked = False

        # Resets example to the start position, can also change sequence size on reset
        def reset(self, sequence_size):
            assert sequence_size is not None

            # Assert that we do not go back more than we go forward
            assert self.offset_size < sequence_size

            self.sequence_size = sequence_size

            self.blocked = False

            # If sample is smaller than sequence size then we simply will ignore it
            margin = self.get_length() - self.sequence_size

            if margin < 0:
                self.done = True
                return

            self.done = False
            self.curr_index = 0

            # Randomly shifts the sequence (Phase)
            if self.random_mode == 1:
                self.curr_index = random.randint(0, min(margin, self.sequence_size))

        # What is uploaded to the info_queue
        def get_info_and_advance(self):
            if self.sequence_size is None:
                raise RuntimeError('Can\'t use example if reset() was not called!')
            if self.done or self.blocked:
                raise RuntimeError('Impossible to get the data for already done or blocked example')

            # When random mode is set to 2 then it ignores the order of sequential chunks and simply reads from a random
            # position. Should be used only when forget_state=True otherwise wrong hidden state will be forwarded.
            if self.random_mode == 2:
                index = np.random.randint(0, self.get_length() - self.sequence_size + 1)
                return self.example_id, (index, self.sequence_size)
            else:
                index = self.curr_index
                self.curr_index += self.sequence_size - self.offset_size

                # If we can't extract more samples then we are done

                self.done = self.curr_index + self.sequence_size > self.get_length()

                self.blocked = True

                return self.example_id, (index, self.sequence_size)

        def get_length(self):
            raise NotImplementedError

        def read_data(self, serialized):
            raise NotImplementedError

    @staticmethod
    def add_arguments(parser):
        parser.section('data_reader')

        parser.add_argument("data_path", type=str,
                            help="Path to the directory containing the data")
        parser.add_argument("readers_count", type=int, default=1,
                            help="Number of parallel data readers.")
        parser.add_argument("batch_size", type=int, default=64,
                            help="Batch size used for training.")
        parser.add_argument("validation_batch_size", type=int, default=0,
                            help="Batch size used for test and validation.")
        parser.add_argument("sequence_size", type=int, default=1000,
                            help="How many time-points are used for each training sequence.")
        parser.add_argument("validation_sequence_size", type=int, default=0,
                            help="Sometimes it might be better/faster to increase the sequence size for validation."
                                 "For example for CNNs.")
        parser.add_argument("balanced", type=int, default=1, choices=[0, 1],
                            help="If greater than 0 then balance mini-batches to have equal number examples per class.")
        parser.add_argument("random_mode", type=int, default=0, choices=[0, 1, 2],
                            help="0 - No randomization; 1 - Reads sequentially but each time starts recording from a "
                                 "new offset; 2 - Reads random chunks, should not be used if forget_state=False."
                                 "Applies only to the train data reader, validation data reader has it set to 0.")
        parser.add_argument("continuous", type=int, default=0, choices=[0, 1],
                            help="If set to 1 then no need to reset after epoch is done.")
        parser.add_argument("limit_examples", type=int, default=0,
                            help="If greater than 0 then will only use this many examples per class.")
        parser.add_argument("limit_duration", type=int, default=0,
                            help="If greater than 0 then each example will only use first limit_duration samples.")
        parser.add_argument("forget_state", type=int, default=0, choices=[0, 1],
                            help="If set to 1 then state will not be forward propagated between subsequences from the "
                                 "same example.")
        parser.add_argument("train_on_full", type=int, default=0, choices=[0, 1], help="If set to 1 then will use "
                                                                                       "all data for training.")
        parser.add_argument("cv_n", type=int, default=5,
                            help="How many folds are used for cross validation.")

        parser.add_argument("cv_k", type=int, default=4,
                            help="Which fold is used for validation. Indexing starts from 0!")
        parser.add_argument("force_parameters", type=int, default=0, choices=[0, 1],
                            help="For Test and validation we change some parameters: balance, forget_state, continuous "
                                 "and random_mode. Setting force_parameters to 1 will disable this change")
        return parser

    def __init__(self,
                 data_path,
                 readers_count,
                 batch_size,
                 validation_batch_size,
                 sequence_size,
                 validation_sequence_size,
                 balanced,
                 random_mode,
                 continuous,
                 limit_examples,
                 limit_duration,
                 forget_state,
                 train_on_full,
                 cv_n,
                 cv_k,
                 force_parameters,
                 offset_size,
                 state_initializer,
                 data_type,
                 allow_smaller_batch,
                 **kwargs):

        assert data_type in self.DataTypes, 'Can not interpret %s as data_type' % data_type

        # Defaults for train
        self.batch_size = batch_size
        self.sequence_size = sequence_size

        if data_type == self.Validation_Data or data_type == self.Test_Data:
            if force_parameters != 1:
                logger.warning('For %s pass we disable: balanced, random_mode, continuous, forget_state' % data_type)
                balanced = False
                random_mode = 0
                continuous = 0
                forget_state = 0

            # If specified by the user then overwrite
            self.batch_size = validation_batch_size if validation_batch_size != 0 else self.batch_size
            self.sequence_size = validation_sequence_size if validation_sequence_size != 0 else self.sequence_size

        self.data_path = data_path

        self.balanced = balanced
        self.random_mode = random_mode
        self.limit_examples = limit_examples
        self.limit_duration = limit_duration
        self.forget_state = forget_state

        self.train_on_full = train_on_full
        self.cv_n = cv_n
        self.cv_k = cv_k
        assert cv_k < cv_n, "Fold used for validation has index which is higher than the number of folds."

        # Offset size is used to adjust hidden state between data chunks (Usually when conv layers are present)
        self.offset_size = offset_size
        # state_initializer function is used to get the initial state (for example: random or zero)
        self.state_initializer = state_initializer

        # Train or Validation type
        self.data_type = data_type
        self.allow_smaller_batch = allow_smaller_batch
        # If set to true then after example is finished it will be immediately reset, so it won't be possible to
        # detect when epoch ends
        self.continuous = continuous

        # Info Queue -> Information that is used to read a proper chunk of data from a proper file
        self.info_queue = multiprocessing.Queue()

        # Data Queue -> Data with labels used for training
        self.data_queue = multiprocessing.Queue()

        # Counts number of samples inside the info_queue
        self.samples_count = 0

        # self.examples[i] -> List with examples for class i
        self.examples = []
        # dictionary example_id -> example for faster access when updating/receiving RNN state
        self.examples_dict = {}

        # We need to update the state between subsequent calls to get_batch()
        self.state_needs_update = False

        # If we are in continuous mode then we will ignore initialize_epoch() calls after the first one was done
        self.epoch_initialized = False

        # Whatever derived class needs
        self._initialize(**kwargs)

        self._create_examples()

        for label, class_examples in enumerate(self.examples):
            logger.debug('Label %s: Number of recordings %d, Cumulative Length %d' %
                         (label, len(class_examples), sum([e.get_length() for e in class_examples])))
        logger.debug('Number of sequences in the dataset %d' % nested_list_len(self.examples))

        # Additional data structure (faster access for some operations)
        for class_examples in self.examples:
            for example in class_examples:
                self.examples_dict[example.example_id] = example

        # Adds up to queue_limit examples at the beginning of each epoch, then after each batch adds batch_size new
        # examples
        self.queue_limit = self.batch_size * 6

        # Create readers
        logger.info('Create reader processes.')
        self.readers = [multiprocessing.Process(target=BaseDataReader.read_sample_function,
                                                args=(self.info_queue, self.data_queue, self.examples_dict))
                        for _ in range(readers_count)]

    # This should be used instead of constructor for the derived class
    # Is this a good design pattern??
    def _initialize(self, **kwargs):
        raise NotImplementedError('Implement instead of constructor')

    def stop_readers(self):
        logger.info('Trying to stop %s readers ...' % self.data_type)

        # There is nothing that adds elements to the info_queue at this point
        # Only readers will try to get elements from this queue.
        while self.info_queue.qsize() > 0:
            try:
                self.info_queue.get(timeout=1)
            except Empty:
                logger.debug('During cleanup, trying to get an element when queue is empty')

        # Will stop readers if they are blocked on the input read
        for _ in range(len(self.readers)):
            self.info_queue.put((None, None))

        # This is super strange to me:
        # If output_queue has more than 1 element then somehow we are not able to join those processes

        # Solution: Simply clear out that data queue to make it possible to nicely shut down
        while self.data_queue.qsize() > 0:
            self.data_queue.get()

        # WHY SOMETIMES IT WILL NOT JOIN THE READERS EVEN THOUGH THEY FINISHED ????????????????
        for (i, r) in enumerate(self.readers):
            logger.debug('Waiting on join for %s reader %d.' % (self.data_type, i))
            try:
                r.join(timeout=1)
                logger.debug('%s reader joined.' % self.data_type.title())
            except TimeoutError:
                logger.warning('Reader %d did not join properly, sending terminate.')
                r.terminate()

    # If info_queue is empty then will stop if receives None as example_id
    @staticmethod
    def read_sample_function(info_queue, output_queue, examples_dict):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        logger.debug('New reader process is running ...')
        while True:
            example_id, serialized = info_queue.get()
            if example_id is not None:
                data = examples_dict[example_id].read_data(serialized)
                output_queue.put(data)

            else:
                logger.debug('Reader received None, finishing the process ...')
                break

        logger.debug('Reader process finished.')

    @staticmethod
    def input_size(**kwargs):
        return NotImplementedError('Trying to call input_size on the BaseDataReader base class')

    @staticmethod
    def output_size(**kwargs):
        return NotImplementedError('Trying to call output_size on the BaseDataReader base class')

    # Implements a method that will fill up self.examples list. This list contains all necessary information about
    # all examples
    def _create_examples(self):
        raise NotImplementedError

    # Probably this function could have lower complexity [Optimize if it takes too much time]
    def _get_random_example(self, class_label=None):
        if class_label is None:
            class_count = len(self.examples)
            all_examples = []
            for class_label in range(class_count):
                all_examples.extend([e for e in self.examples[class_label] if (not e.done) and (not e.blocked)])

        else:
            # Find all not done and not blocked examples for this class
            all_examples = [e for e in self.examples[class_label] if (not e.done) and (not e.blocked)]

        if len(all_examples) == 0:
            return None

        v = random.choice(all_examples)
        return v

    # This should append up to 'size' samples to the info_queue. We make sure that at a given point in time at most
    # one sequence from each example is inside the info queue and data_queue
    def _append_samples(self, size):
        if self.balanced:
            class_count = len(self.examples)
            # Size has to be N * number_of_classes, where N is an integer
            assert size % class_count == 0

            # Extract s samples per class
            s = size // class_count
            for i in range(s):
                # Get info about random example for each class
                examples = [self._get_random_example(class_label) for class_label in range(class_count)]
                if None in examples:
                    return

                # Put info about random examples to the queue
                for example in examples:
                    self.samples_count += 1
                    self.info_queue.put(example.get_info_and_advance())

        else:
            for _ in range(size):
                # Get a random example from any class
                example = self._get_random_example(class_label=None)
                if example is None:
                    return

                self.samples_count += 1
                self.info_queue.put(example.get_info_and_advance())

    def initialize_epoch(self, sequence_size=None):
        if (self.data_type != self.Train_Data) and self.random_mode:
            logger.warning('Are you sure you want to set random_mode in {1, 2} for non training data?')

        if self.continuous and self.epoch_initialized:
            logger.warning('Trying to initialize a new epoch (%s) but mode is set to continuous, skipping'
                           % self.data_type)
            return

        if sequence_size is None:
            sequence_size = self.sequence_size

        if sequence_size != self.sequence_size:
            logger.debug('Changing sequence size from %d to %d ' % (self.sequence_size, sequence_size))
            self.sequence_size = sequence_size

        logger.debug('Initialize new epoch (%s)' % self.data_type)

        # Read examples that we were unable to process in the previous epoch
        for i in range(self.samples_count):
            self.data_queue.get()
            self.samples_count -= 1

        for class_examples in self.examples:
            for example in class_examples:
                example.reset(sequence_size=self.sequence_size)
                example.state = self.state_initializer()

        # Populate info queue with some examples, up to queue_limit
        self._append_samples(self.queue_limit)

        self.state_needs_update = False
        self.epoch_initialized = True

    def set_states(self, keys, states):
        assert(len(keys) == len(states))

        for key, state in zip(keys, states):
            # If continuous training (no epochs) then we need to reset example if it was finished (done)
            if self.continuous and self.examples_dict[key].done == True:
                self.examples_dict[key].reset(sequence_size=self.sequence_size)
                self.examples_dict[key].state = self.state_initializer()
            else:
                self.examples_dict[key].state = state
                self.examples_dict[key].blocked = False

        self._append_samples(self.batch_size)
        self.state_needs_update = False

    def get_states(self, keys):
        states = []
        for key in keys:
            if self.forget_state:
                states.append(self.state_initializer())
            else:
                states.append(self.examples_dict[key].state)
        return states

    def start_readers(self):
        logger.debug('Starting %s readers.' % self.data_type)
        for r in self.readers:
            r.start()

    def get_batch(self):
        if self.state_needs_update:
            logger.error('State needs an update.')
            raise RuntimeError('State needs an update.')
        if self.epoch_initialized is False:
            logger.error('Trying to get batch while epoch is not initialized.')
            raise RuntimeError('Epoch not initialized.')

        data_arrays = []
        time_arrays = []
        labels = []
        ids = []
        context_arrays = []

        if self.samples_count < self.batch_size:
            if not self.allow_smaller_batch or self.samples_count == 0:
                raise BaseDataReader.EpochDone
            else:
                batch_size = self.samples_count
        else:
            batch_size = self.batch_size

        for i in range(batch_size):
            data, time, label, example_id, context = self.data_queue.get()
            self.samples_count -= 1
            data_arrays.append(data)
            time_arrays.append(time)
            labels.append(label)
            ids.append(example_id)
            context_arrays.append(context)

        self.state_needs_update = True
        return ids, np.stack(data_arrays, axis=0), np.stack(time_arrays, axis=0), np.stack(labels, axis=0), \
               np.stack(context_arrays, axis=0)

    def cv_split(self, data_list):
        """
        Helper function that based on values set for cv_k and cv_n will split the data into train and validation set.
        """
        size = len(data_list)
        assert size != 0, 'Can not make a cv split from an empty data list.'

        # If data_type is set to test no split will be made.
        if self.data_type == self.Test_Data:
            return data_list

        # If we want to train on the full training dataset for final evaluation then return all the data
        if self.train_on_full:
            assert self.data_type != self.Validation_Data, 'Can not use validation if train_on_full is set to 1'
            return data_list

        # Split out the data according to the CV fold
        start = int(self.cv_k/self.cv_n * size)
        end = int((self.cv_k+1)/self.cv_n * size)

        logger.debug("Using CV split cv_n: %s, cv_k: %s, start: %s, end: %s" % (self.cv_n, self.cv_k, start, end))

        if self.data_type == self.Train_Data:
            return data_list[:start] + data_list[end:]
        else:
            return data_list[start:end]
