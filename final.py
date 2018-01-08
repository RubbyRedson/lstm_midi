import numpy as np
import tensorflow as tf
import time

VERBOSE = True


class Config(object):
    data_path = '/tmp/chopin/data'
    save_dir = '/tmp/chopin/save'
    seed_str = ' 58%,1 58%,1 58%,1'
    num_epochs = 100
    num_units = 128
    batch_size = 16
    seq_len = 128
    num_layers = 3
    dropout = 1.0
    learning_rate = 1e-3
    save_epochs = 5

    @property
    def save_path(self):
        import os
        filename = "xmas-{}units_{}epochs_{}batchsize_{}seqlen_{}layers_{}dropout_{}lr".format(
            self.num_units, self.num_epochs, self.batch_size, self.seq_len, self.num_layers, self.dropout,
            self.learning_rate
        )
        return os.path.join(self.save_dir, filename)

    def save_condition(self, *args):
        return False


class DefaultConfig(Config):
    data_path = './midi_text/bach/pop.txt'
    save_dir = './charbased/saves/'

    def save_condition(self, **kwargs):
        if 'epoch' in kwargs.keys():
            epoch = kwargs['epoch']
            return epoch == 1  or epoch % 5 == 0

class PredictConfig(Config):
    batch_size = 1
    seq_len = 1
    

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

        # self.vocab = set(raw_data)
        # self.vocab_size = len(self.vocab)
        # self.idx_to_char = dic = dict(enumerate(self.vocab))
        import dictionary
        self.idx_to_char = dic = dictionary.getDict()
        self.char_to_idx = rev = dictionary.getDictSwapped()

        self.vocab = set(self.char_to_idx.keys())
        self.vocab_size = len(self.vocab)

        # data = [rev[c] for c in raw_data]
        data = []
        blacklist = set()
        blacklisted = 0
        for c in raw_data:
            if c in rev:
                data.append(rev[c])
            else:
                blacklist.add(c)
                blacklisted += 1
        print("INFO: Blacklisted {} ({} total) characters from input data.".format(len(blacklist), blacklisted))
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

        # Reuse variables to allow multiple concurrent models (for training and sampling, for ex.)
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE) as scope:
            if self.interactive:
                scope.reuse_variables()

            self.inputs = inputs = tf.placeholder(tf.int32, [config.batch_size, config.seq_len], name='inputs')
            self.labels = labels = tf.placeholder(tf.int32, [config.batch_size, config.seq_len], name='labels')

            use_dropout = self.training and config.dropout < 1.0
            is_multilayer = config.num_layers > 1

            dropout = tf.constant(config.dropout, name="dropout")

            # TODO Swap out embeddings for manual OHE
            embeddings = tf.get_variable('embeddings', [config.vocab_size, config.num_units])
            rnn_inputs = tf.nn.embedding_lookup(embeddings, self.inputs)

            self.cell = cell = tf.nn.rnn_cell.BasicLSTMCell(config.num_units)
            if use_dropout:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout)
            if is_multilayer:
                cell = tf.nn.rnn_cell.MultiRNNCell([cell] * config.num_layers)

            if use_dropout:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout)

            # Unroll LSTM cells
            self.init_state = init_state = cell.zero_state(config.batch_size, tf.float32)
            rnn_outputs, last_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
            self.final_state = last_state

            w_initializer = tf.constant_initializer(0.0)
            b_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.8)
            W = tf.get_variable('W', [config.num_units, config.vocab_size], initializer=w_initializer)
            b = tf.get_variable('b', [config.vocab_size], initializer=b_initializer)

            # Flatten outputs
            rnn_outputs = tf.reshape(rnn_outputs, [-1, config.num_units])
            labels_out = tf.reshape(labels, [-1])

            # self.logits = tf.nn.xw_plus_b(rnn_outputs, W, b)
            self.logits = tf.matmul(rnn_outputs, W) + b
            self.predictions = tf.nn.softmax(self.logits)

            pred_indices = tf.argmax(self.predictions, 1)
            correct = tf.equal(pred_indices, tf.cast(labels_out, pred_indices.dtype))
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=labels_out))
            self.train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.loss)
            # TODO Gradient clipping?

            self.init_op = tf.global_variables_initializer()
            # TODO Make sample() function for evaluating prediction against actual

    def sample(self, sess, idx_to_char, char_to_idx, num=256, seed='The ', sampling_type="weighted"):
        def weighted_pick(p):
            idx = np.random.choice(range(loader.vocab_size), p=p.ravel())
            return idx

        # Verify seed
        for c in seed:
            if c not in char_to_idx:
                import random
                seed_ = ''.join([random.choice(list(char_to_idx.keys())) for _ in range(len(seed))])
                print("WARN: Seed is not compatible with dictionary. Using random seed '{}'".format(seed_))
                seed = seed_
                break

        # Warm up LSTM units
        state = sess.run(self.init_state)
        for char in seed[:-1]:
            cur = char_to_idx[char]
            feed = {
                self.inputs: [[cur]],
            }
            if state is not None:
                feed[self.init_state] = state

            state = sess.run(self.final_state, feed)

        ret = seed
        char = seed[-1]
        n = 0
        num_spaces = 0

        while (n < num * 2) and (num_spaces < num):
            n += 1
            cur = char_to_idx[char]

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

            pred = idx_to_char[sample]
            if pred == ' ':
                num_spaces += 1
            ret += pred
            char = pred
        return ret


def train(config, loader):
    model = Model(config, training=True)
    predict_conf = PredictConfig()
    predict_conf.vocab_size = config.vocab_size

    predict_model = Model(predict_conf, training=False, interactive=True)
    losses = []

    with tf.Session() as sess:
        sess.run(model.init_op)
        sess.run(predict_model.init_op)
        saver = tf.train.Saver()
        counter = 0

        for idx, epoch in enumerate(loader.gen_epochs(config.num_epochs)):
            # Reset memory
            loss = 0
            acc = 0
            steps = 1
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

                l, a, state, _ = sess.run([
                    model.loss,
                    model.accuracy,
                    model.final_state,
                    model.train_op
                ], feed_dict=feed)


                loss += l
                acc += a
                counter += 1

                if counter % 1000 == 0:
                    print("Total counter currently at {}".format(counter))

            end = time.time()

            print("({}s) Epoch {}/{}: avg. loss={}, avg. acc={}".format(end - start, idx, config.num_epochs, loss / steps, acc / steps))
            losses.append(loss / steps)

            sentence = predict_model.sample(sess, loader.idx_to_char, loader.char_to_idx, sampling_type="weighted", seed=config.seed_str)
            print(sentence)

            if config.save_condition(epoch=idx):
                print("Saving to {}".format(config.save_path + '-' + str(idx)))
                saver.save(sess, config.save_path + '-' + str(idx))
        print("[DONE] Saving to {}".format(config.save_path + '-final'))
        saver.save(sess, config.save_path + '-final')
    return losses


def predict(config, loader, save_path=None):
    if save_path is None:
        save_path = config.save_path
    model = Model(config, training=False)
    with tf.Session() as sess:
        tf.train.Saver().restore(sess, save_path)
        sentence = model.sample(sess, loader.idx_to_char, loader.char_to_idx, sampling_type="weighted")
        print(sentence)


if __name__ == '__main__':
    config = DefaultConfig()
    loader = DataLoader(config)
    config.vocab_size = loader.vocab_size

    train(config, loader)
    # TODO By training, it seems more believable. by reading savepoint, it's garbled and doesn't seem to have learned
    # TODO Pickle data in loader (like vocab_size and dic, rev)
    # predict(config, loader, save_path='charbased/saves/128units_50epochs_16batchsize_25seqlen_2layers_0.7dropout_0.001lr')
