import tensorflow as tf
from src.dl_core import ModelTrainerBase


class ModelTrainer(ModelTrainerBase):
    def __init__(self, model, learning_rate, weight_decay, train_dr, test_dr, sequence_size, loss_type):
        super().__init__(model, learning_rate, weight_decay, train_dr, test_dr, loss_type)

        self.global_step = tf.Variable(0, trainable=False)

        self.input_placeholder = tf.placeholder(tf.float32,
                                                shape=[None, sequence_size, self.train_dr.input_dim],
                                                name='input_placeholder')
        self.time_placeholder = tf.placeholder(tf.float32,
                                               shape=[None, sequence_size, 1],
                                               name='time_placeholder')
        self.target_placeholder = tf.placeholder(tf.int32,
                                                 shape=[None, sequence_size],
                                                 name='target_placeholder')

        self.state_placeholder, self.state = self.model.state_placeholders()
        self.forward_op = self.model.forward((self.time_placeholder, self.input_placeholder), self.state)

        output, state = self.forward_op

        if self.loss_type == 'classification_last':
            training_outputs = output[:, -1, :]
            training_labels = self.target_placeholder[:, -1]
        elif self.loss_type == 'classification_all':
            outputs_num = int(output.get_shape()[-1])
            training_outputs = tf.reshape(output, shape=[-1, outputs_num])
            training_labels = tf.reshape(self.target_placeholder, [-1])
        else:
            raise NotImplementedError

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=training_outputs,
                                                                                  labels=training_labels))

        # L2 Normalization
        tv = tf.trainable_variables()
        for v in tv:
            print(v.name)

        self.reg_los = tf.reduce_sum([tf.nn.l2_loss(v) for v in tv if 'bias' not in v.name])

        self.loss_all = self.loss + self.weight_decay * self.reg_los
        self.optimization_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_all, self.global_step)

        self.sess = tf.InteractiveSession()

        self.sess.run(tf.global_variables_initializer())

    def get_placeholder_dict(self, state):
        d = dict()
        for p, s in zip(self.state_placeholder, state):
            d[p] = s

        return d

    def _one_iteration(self, batch, time, hidden, labels, update=False):
        ops = [self.forward_op, self.loss]

        if update:
            ops.append(self.optimization_op)

        feed_dict = self.get_placeholder_dict(hidden)
        feed_dict[self.time_placeholder] = time
        feed_dict[self.input_placeholder] = batch
        feed_dict[self.target_placeholder] = labels

        r = self.sess.run(ops, feed_dict)

        (prediction, hidden), loss = r[:2]

        return prediction, hidden, loss

    def _gather_results(self, ids, outputs, labels, loss):
        self.metrics.append_results(ids, outputs, labels, loss)

