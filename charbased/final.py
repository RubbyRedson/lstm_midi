import numpy as np
import tensorflow as tf
import time

VERBOSE = True


class Config(object):
    data_path = '/tmp/chopin/data'
    save_dir = '/tmp/chopin/save'
    num_epochs = 30
    num_units = 128
    batch_size = 16
    seq_len = 25
    num_layers = 3
    dropout = 0.7
    learning_rate = 1e-3

    @property
    def save_path(self):
        import os
        filename = "{}units_{}epochs_{}batchsize_{}seqlen_{}layers_{}dropout_{}lr".format(
            self.num_units, self.num_epochs, self.batch_size, self.seq_len, self.num_layers, self.dropout, self.learning_rate
        )
        return os.path.join(self.save_dir, filename)


class DefaultConfig(Config):
    data_path = './charbased/example/input.txt'
    save_dir = './charbased/saves/'


class Parser(object):
    def parse(self, data_path):
        raise NotImplementedError


class BasicCharParser(Parser):
    def parse(self, data_path):
        # TODO handle directory vs file
        raw_data = open(data_path, 'r').read()
        return raw_data


class DataLoader(object):

    def __init__(self, config):
        self.load(config.data_path, config.batch_size, config.seq_len)

    def load(self, data_path, batch_size, seq_len, encoding='utf-8', parser=None):
        if parser is None:
            print("No parser defined, resorting to Char-Based")
            parser = BasicCharParser()
        self.data_path = data_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.encoding = encoding
        self.parser = parser

        raw_data = parser.parse(data_path)

        self.vocab = set(raw_data)
        self.vocab_size = len(self.vocab)
        self.dic = dic = dict(enumerate(self.vocab))
        self.rev = rev = dict(zip(dic.values(), dic.keys()))

        data = [rev[c] for c in raw_data]
        self.data = np.array(data, dtype=np.int32)
        del raw_data

    def batch_iter(self):
        assert self.data is not None

        data_len = len(self.data)
        batch_len = data_len // self.batch_size

        data = np.zeros([self.batch_size, batch_len], dtype=np.int32)
        offset = 0

        if data_len % self.batch_size:
            offset = np.random.randint(0, data_len % self.batch_size)
        for i in range(self.batch_size):
            data[i] = self.data[batch_len * i + offset:batch_len * (i + 1) + offset]

        epoch_size = (batch_len - 1) // self.seq_len

        if epoch_size == 0:
            raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

        for i in range(epoch_size):
            x = data[:, i * self.seq_len: (i + 1) * self.seq_len]
            y = data[:, i * self.seq_len + 1: (i + 1) * self.seq_len + 1]

            yield (x, y)

    def gen_epochs(self, epochs):
        for i in range(epochs):
            yield self.batch_iter()


class Model(object):
    def __init__(self, config, training=True, interactive=False):
        self.config = config
        self.training = training
        self.interactive = interactive
        if not training:
            config.batch_size = 1
            config.seq_len = 1

        self.generate()

    def generate(self):
        assert self.config is not None
        config = self.config

        if VERBOSE:
            print("Generating graph")

        with tf.variable_scope("model", reuse=tf.AUTO_REUSE) as scope:
            if self.interactive:
                scope.reuse_variables()

            self.inputs = inputs = tf.placeholder(tf.int32, [config.batch_size, config.seq_len], name='inputs')
            self.labels = labels = tf.placeholder(tf.int32, [config.batch_size, config.seq_len], name='labels')

            use_dropout = self.training and config.dropout < 1.0
            is_multilayer = config.num_layers > 1

            dropout = tf.constant(config.dropout, name="dropout")

            embeddings = tf.get_variable('embeddings', [config.vocab_size, config.num_units])
            rnn_inputs = tf.nn.embedding_lookup(embeddings, self.inputs)

            self.cell = cell = tf.nn.rnn_cell.BasicLSTMCell(config.num_units)
            if use_dropout:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout)
            if is_multilayer:
                cell = tf.nn.rnn_cell.MultiRNNCell([cell] * config.num_layers)

            if use_dropout:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout)

            self.init_state = init_state = cell.zero_state(config.batch_size, tf.float32)
            rnn_outputs, last_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
            self.final_state = last_state

            initializer = tf.constant_initializer(0.0)
            # initializer = tf.random_normal_initializer(mean=0.0, stddev=0.8)
            with tf.variable_scope('rnnlm'):
                W = tf.get_variable('W', [config.num_units, config.vocab_size])
                b = tf.get_variable('b', [config.vocab_size], initializer=initializer)

            # Reshape outputs (we only really care about the last unit's predictions for the last char)
            rnn_outputs = tf.reshape(rnn_outputs, [-1, config.num_units])
            labels_out = tf.reshape(labels, [-1])

            # self.logits = tf.nn.xw_plus_b(rnn_outputs, W, b)
            self.logits = tf.matmul(rnn_outputs, W) + b
            self.predictions = tf.nn.softmax(self.logits)

            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=labels_out))
            self.train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.loss)

            self.init_op = tf.global_variables_initializer() if not self.interactive else None
            # TODO Make sample() function for evaluating prediction against actual

    def sample(self, sess, rev, dic, num=200, seed='The ', sampling_type="weighted"):
        def weighted_pick(p):
            idx = np.random.choice(range(loader.vocab_size), p=p.ravel())
            return idx

        # Warm up LSTM units
        state = sess.run(self.init_state)
        for char in seed[:-1]:
            cur = rev[char]
            feed = {
                self.inputs: [[cur]],
            }
            if state is not None:
                feed[self.init_state] = state

            state = sess.run(self.final_state, feed)

        ret = seed
        char = seed[-1]
        for n in range(num):
            cur = rev[char]

            feed = {self.inputs: [[cur]], self.init_state: state}
            [probs, state] = sess.run([self.predictions, self.final_state], feed)
            p = probs[0]

            if sampling_type == "argmax":
                sample = np.argmax(p)
            elif sampling_type == "hybrid":
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else:
                sample = weighted_pick(p)

            pred = dic[sample]
            ret += pred
            char = pred
        return ret


def train(config, loader):
    model = Model(config, training=True)
    predict_model = Model(config, training=False, interactive=True)
    losses = []

    with tf.Session() as sess:
        sess.run(model.init_op)
        saver = tf.train.Saver()

        for idx, epoch in enumerate(loader.gen_epochs(config.num_epochs)):
            # Reset memory
            loss = 0
            steps = 0
            state = sess.run(model.init_state)
            start = time.time()

            for X, Y in epoch:
                steps += 1
                feed = {
                    model.inputs: X,
                    model.labels: Y,
                }
                if state is not None:
                    feed[model.init_state] = state

                l, state, _ = sess.run([
                    model.loss,
                    model.final_state,
                    model.train_op
                ], feed_dict=feed)

                loss += l

                if steps % 1000 == 0:
                    sentence = predict_model.sample(sess, loader.rev, loader.dic, sampling_type="weighted")
                    print(sentence)

            end = time.time()

            print("({}s) Epoch {}/{}: avg. loss={}".format(end - start, idx, config.num_epochs, loss / steps))
            losses.append(loss / steps)

        print("Saving to {}".format(config.save_dir))
        saver.save(sess, config.save_dir)
    return losses


def predict(config, loader):
    model = Model(config, training=False)
    with tf.Session() as sess:
        tf.train.Saver().restore(sess, config.save_dir)
        sess.run(model.init_op)
        sentence = model.sample(sess, loader.rev, loader.dic, sampling_type="weighted")
        print(sentence)


if __name__ == '__main__':
    config = DefaultConfig()
    loader = DataLoader(config)
    config.vocab_size = loader.vocab_size

    import pprint

    pprint.pprint(config)

    train(config, loader)
    # TODO By training, it seems more believable. by reading savepoint, it's garbled and doesn't seem to have learned
    # TODO Pickle data in loader (like vocab_size and dic, rev)
    predict(config, loader)
