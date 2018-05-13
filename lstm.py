from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
from datetime import datetime
from dataLoader import DataLoader
from model import Model
import numpy as np


# Parameters
learning_rate = 0.001
training_iters = -1
display_step = 1000
n_input = 36
# number of units in RNN cell
n_hidden = 256
minibatch_size = 32
dropout = 0.95
n_layers = 4

# Target log path
SESSION_NAME = "{}-Layers_{}-LR_{}-mem_{}-units".format(n_layers, learning_rate, n_input, n_hidden)
logs_path = "/logs/training/{}".format(SESSION_NAME)

training_folder = './midi_text'
#training_folder = './cello_text'
save_loc = './resources/models/model.ckpt'
loader = DataLoader(n_input, minibatch_size, training_folder, None)
loader.loadData(True)
vocab_size = len(loader.dictionary)

model = Model(n_hidden, n_input, n_layers, vocab_size)

def parse_function(example_proto):
	features = {'train/feature': tf.FixedLenFeature([n_input, 1], tf.float32),
			   'train/label': tf.FixedLenFeature([vocab_size], tf.float32)}

	parsed_features = tf.parse_single_example(example_proto, features)
	return parsed_features["train/feature"], parsed_features["train/label"]


dataset = tf.data.Dataset.from_tensor_slices((loader.batches, loader.labels))
#dataset = dataset.map(parse_function)
dataset = dataset.prefetch(1000).repeat().batch(minibatch_size)

def toOneHot(features, labels):
	_y = tf.one_hot(labels, vocab_size, axis=1)
	_y = tf.reshape(_y, [-1, vocab_size])
	return features, _y



#dataset = tf.data.Dataset.from_tensor_slices(loader.training_data)
#dataset = dataset.map(toOneHot).prefetch(1000).repeat().batch(minibatch_size)

iter = dataset.make_one_shot_iterator()
x, y = iter.get_next()
model.x = x
model.y = y


pred = model.RNN()

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=model.y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(model.y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# For keeping track of iterations in the session
iteration = tf.Variable(0, name="iteration")

# Initializing the variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()

#Create a variable scope for the logging
with tf.variable_scope('logging'):
	tf.summary.scalar('loss', cost)
	tf.summary.scalar('accuracy', accuracy)

with tf.name_scope("model"):
	tf.summary.histogram("weights", model.weights['out'])
	tf.summary.histogram("biases", model.biases['out'])
	summary = tf.summary.merge_all()

# Launch the graph
with tf.Session() as session:
	session.run(init)
	#saver.restore(session, save_loc)
	step = iteration.eval()
	acc_total = 0
	loss_total = 0

	training_writer = tf.summary.FileWriter('./logs/{}/training'.format(SESSION_NAME), session.graph)

	while step < training_iters or training_iters < 0:
		try:

			_, acc, loss, training_summary = session.run([optimizer, accuracy, cost, summary], feed_dict={model.pkeep: dropout})

			loss_total += loss
			acc_total += acc
			training_writer.add_summary(training_summary, step)

			if (step+1) % display_step == 0:
				#onehot_pred = session.run(pred, feed_dict={model.pkeep: 1.0})

				print("[{}-trainingset] Iter= ".format(str(datetime.now())) + str(step + 1) + ", Average Loss= " + \
					  "{:.6f}".format(loss_total / display_step) + ", Average Accuracy= " + \
					  "{:.2f}%".format(100 * acc_total / display_step))

				
				session.run(iteration.assign(step))
				saver.save(session, save_loc)
				acc_total = 0
				loss_total = 0
				
				'''
				symbols_in = [loader.reverse_dictionary[i[0]] for i in symbols_in_keys[0]]
				symbols_out = loader.reverse_dictionary[int(tf.argmax(symbols_out_onehot[0], 0).eval())]
				symbols_out_pred = loader.reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval()[0])]

				print("%s - [%s] vs [%s]" % (symbols_in, symbols_out, symbols_out_pred))
				'''

			step += 1
		except KeyboardInterrupt:
			print("Exiting and saving model")
			session.run(iteration.assign(step))
			break

	saver.save(session, save_loc)
