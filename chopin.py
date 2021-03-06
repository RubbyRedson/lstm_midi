import os
import random
import sys
import time
import numpy as np
import tensorflow as tf
from dictionary import getDictSwapped
from tensorflow.contrib import rnn
import collections


# Constants
# DEFAULT_DATA_PATH = './midi_text/midi_text/bach/bach_846_format0_track_all.txt'
DEFAULT_DATA_PATH = './charbased/example/input.txt'
DEFAULT_LOGDIR = './resources'
DEFAULT_SAVE_LOC = './resources/model.ckpt'

def elapsed(sec):
    if sec < 60:
        return str(sec) + " sec"
    elif sec < (60 * 60):
        return str(sec / 60) + " min"
    else:
        return str(sec / (60 * 60)) + " hr"


def read_file(file_path):
    """
    Consumes a given textfile by stripping and splitting it into words
    :param file_path: location of file to read
    :return: numpy array of the file's words
    """
    # if "track_all.txt" not in file_path:
    #     return np.array([])


    print("Parsing {}".format(file_path))

    with open(file_path) as f:
        # lines = f.readlines()
        data = f.read()
        return data



def read_dir(dir_path):
    """
    Consumes all textfiles in a directory by stripping and splitting them into words
    :param dir_path: location of directory to read
    :return: list of numpy arrays of files' words
    """
    contents = [read_data(os.path.join(dir_path, f)) for f in os.listdir(dir_path)]
    return contents

def read_data(data_path):
    """
    Consumes all data files at the given path
    :param data_path: data to read
    :return: list of numpy arrays of words
    """
    if os.path.isfile(data_path):
        return [read_file(data_path)]
    elif os.path.isdir(data_path):
        return read_dir(data_path)


# TODO I don't think we need this, as our dictionary is finite and known ahead of time
def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary
    #
    #
    # dictionary = getDictSwapped()
    # reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    #
    # return dictionary, reverse_dictionary


def get_lstm_cell(num_hidden):
    return rnn.BasicLSTMCell(num_hidden)


def create_rnn(x, weights, biases, num_inputs, num_hidden, pkeep):
    def make_cell():
        cell = get_lstm_cell(num_hidden)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=pkeep)

        return cell

    x = tf.reshape(x, [-1, num_inputs])

    # Split into n-element sequences of inputs
    x = tf.split(x, num_inputs, 1)

    num_layers = 2

    rnn_cell = rnn.MultiRNNCell([make_cell() for _ in range(num_layers)])

    # Generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are num_inputs outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


def run(data_path=DEFAULT_DATA_PATH, logdir=DEFAULT_LOGDIR, save_loc=DEFAULT_SAVE_LOC):
    start_time = time.time()

    # Parameters
    learning_rate = 0.0001
    training_iters = 2000
    display_step = 100
    n_input = 100
    n_predictions = 200
    n_hidden = 512
    _pkeep = 0.7


    # Consume data files and build representation
    training_data = read_file(data_path)
    #print("training data: {}".format(training_data))
    # Flatten into single array
    # training_data = np.concatenate(training_data).ravel()
    # training_data = [element for tupl in training_data for element in tupl]

    dictionary, reverse_dictionary = build_dataset(training_data)

    vocab_size = len(dictionary)
    print(dictionary)

    writer = tf.summary.FileWriter(logdir=logdir)

    x = tf.placeholder("float", [None, n_input, 1])
    y = tf.placeholder("float", [None, vocab_size])
    pkeep = tf.placeholder(tf.float32)

    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, vocab_size], mean=0.0, stddev=0.08))
    }
    # TODO Initialize LSTM forget gates with higher biases to encourage remembering in beginning
    biases = {
        'out': tf.Variable(tf.random_normal([vocab_size], mean=0.0, stddev=0.8))
    }

    logits = create_rnn(x, weights, biases, n_input, n_hidden, pkeep)
    pred = prediction = tf.nn.softmax(logits)

    # Loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cost, global_step=tf.train.get_or_create_global_step())

    # Model evaluation
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Launch TF
    with tf.Session() as session:
        session.run(init)
        step = 0
        offset = random.randint(0, n_input + 1)
        end_offset = n_input + 1
        acc_total = 0
        loss_total = 0

        writer.add_graph(session.graph)

        while step < training_iters:
            if offset > (len(training_data) - end_offset):
                # If we've stepped past our input data, restart at random offset
                offset = random.randint(0, n_input + 1)


            symbols_in_keys = [[dictionary[str(training_data[i])]] for i in range(offset, offset + n_input)]
            symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])

            symbols_out_onehot = np.zeros([vocab_size], dtype=float)
            symbols_out_onehot[dictionary[str(training_data[offset + n_input])]] = 1.0
            symbols_out_onehot = np.reshape(symbols_out_onehot, [1, -1])

            _, acc, loss, onehot_pred = session.run(
                [train_op, accuracy, cost, pred],
                feed_dict={x: symbols_in_keys, y: symbols_out_onehot, pkeep: _pkeep})


            loss_total += loss
            acc_total += acc
            if (step + 1) % display_step == 0:
                print("Iter= " + str(step + 1) + ", Average Loss= " +
                      "{:.6f}".format(loss_total / display_step) + ", Average Accuracy= " +
                      "{:.2f}%".format(100 * acc_total / display_step))
                acc_total = 0
                loss_total = 0
                symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
                symbols_out = training_data[offset + n_input]
                symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
                print("%s - [%s] vs [%s]" % (symbols_in, symbols_out, symbols_out_pred))
            if (step + 1) % 10000 == 0:
                save_path = saver.save(session, save_loc)
                print("Model saved in file: %s" % save_path)

                coll = ""
                prime = [[dictionary[str(training_data[i])]] for i in range(offset, offset + n_input)]
                for j in range(100):
                    xin =  np.reshape(np.array(prime), [-1, n_input, 1])
                    p = session.run(pred, {x: xin})
                    idx = np.argmax(np.random.multinomial(1, pvals=p))
                    char = dictionary[idx]
                    coll += char
                print(coll)

            step += 1
            offset += (n_input + 1)


        print("Optimization Finished!")
        print("Elapsed time: ", elapsed(time.time() - start_time))
        save_path = saver.save(session, save_loc)
        print("Model saved in file: %s" % save_path)

        print("Run on command line.")
        print("\ttensorboard --logdir=%s" % logdir)
        print("Point your web browser to: http://localhost:6006/")

        while True:
            prompt = "%s words: " % n_input
            sentence = input(prompt)

            words = [c for c in sentence]
            print(words)
            #sentence = sentence.strip()
            #words = sentence.split(' ')
            if len(words) != n_input:
                continue
            try:
                symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
                print(symbols_in_keys)

                predictionOutput = ""
                for i in range(n_predictions):
                    keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                    onehot_pred = session.run(pred, feed_dict={x: keys, pkeep: 1})


                    # onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                    onehot_pred_index = int(np.searchsorted(np.cumsum(onehot_pred), np.random.rand(1)))
                    sentence = "%s%s" % (sentence, reverse_dictionary[onehot_pred_index])
                    print("Prediction" + reverse_dictionary[onehot_pred_index] + "INDEX:" + str(onehot_pred_index))
                    symbols_in_keys = symbols_in_keys[1:]
                    predictionOutput += reverse_dictionary[onehot_pred_index]
                    symbols_in_keys.append(onehot_pred_index)

                print(sentence)
                print(predictionOutput)
            except:
                print("Word not in dictionary")


if __name__ == '__main__':
    data_path = DEFAULT_DATA_PATH if len(sys.argv) is 1 else sys.argv[1]
    run(data_path)
