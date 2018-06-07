from tensorflow.contrib.rnn import BasicLSTMCell, GRUCell, PhasedLSTMCell, LayerNormBasicLSTMCell
from src.deep_learning.tensorflow import MultiPhasedLSTMCell
from tensorflow.python.util.nest import flatten
from tensorflow.contrib.layers import fully_connected
import tensorflow as tf
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SimpleRNN:
    cell_mapper = {
        'LSTM': BasicLSTMCell,
        'GRU': GRUCell,
        'PhasedLSTM': PhasedLSTMCell,
        'LayerNormLSTM': LayerNormBasicLSTMCell
    }

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout, cell_type):
        logger.warn('DROPOUT NOT IMPLEMENTED')

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.cell_type = cell_type
        self.cells = [self.cell_mapper[cell_type](num_units=self.hidden_size) for _ in range(self.num_layers)]

        if cell_type == 'PhasedLSTM':
            self.network = MultiPhasedLSTMCell(self.cells)
        else:
            self.network = tf.contrib.rnn.MultiRNNCell(self.cells)

    def forward(self, x, hidden):
        if self.cell_type != 'PhasedLSTM':
            time, x = x
        outputs, hidden = tf.nn.dynamic_rnn(self.network, x, initial_state=hidden, dtype=tf.float32)
        shape = outputs.get_shape()
        outputs = tf.reshape(outputs, [-1, shape[2]])
        outputs = fully_connected(outputs, num_outputs=2, activation_fn=None)
        outputs = tf.reshape(outputs, [-1, shape[1], 2])

        return outputs, hidden

    def initial_state(self):
        state = []

        if self.cell_type in ['LSTM', 'PhasedLSTM', 'LayerNormLSTM']:
            for _ in range(self.num_layers):
                c = np.random.normal(0, 1.0, [1, self.hidden_size])
                h = np.random.normal(0, 1.0, [1, self.hidden_size])
                state.extend([c, h])

        if self.cell_type in ['GRU']:
            for _ in range(self.num_layers):
                c = np.random.normal(0, 1.0, [1, self.hidden_size])
                state.append(c)
        return state

    def state_placeholders(self):
        state = []
        phs = []
        if self.cell_type in ['LSTM', 'PhasedLSTM', 'LayerNormLSTM']:
            for _ in range(self.num_layers):
                c = tf.placeholder(dtype=tf.float32, shape=[None, self.hidden_size])
                h = tf.placeholder(dtype=tf.float32, shape=[None, self.hidden_size])
                phs.append(c)
                phs.append(h)
                state.append(tf.contrib.rnn.LSTMStateTuple(c, h))
        elif self.cell_type in ['GRU']:
            for _ in range(self.num_layers):
                c = tf.placeholder(dtype=tf.float32, shape=[None, self.hidden_size])
                phs.append(c)
                state.append(c)
        else:
            raise NotImplementedError

        return phs, tuple(state)

    @staticmethod
    def export_state(states):
        flatten_states = flatten(states)
        batch_size = flatten_states[0].shape[0]

        exported_states = []
        for i in range(batch_size):
            exported_states.append([f_s[np.newaxis, i] for f_s in flatten_states])

        return exported_states

    @staticmethod
    def import_state(states):
        state_tensor_count = len(states[0])
        imported_states = []
        for i in range(state_tensor_count):
            imported_states.append(np.concatenate([s[i] for s in states]))

        return tuple(imported_states)

    def save_model(self, path):
        pass

    def load_model(self, path):
        pass

    @staticmethod
    def count_params():
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters
#
#
# class PhasedLSTM(TensorflowModel):
#     def __init__(self):
#         pass
#
#     def forward(self, x, hidden):
#         pass
