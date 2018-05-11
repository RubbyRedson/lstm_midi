from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
from datetime import datetime
from dataLoader import DataLoader

# Parameters
learning_rate = 0.001
training_iters = -1
display_step = 1000
n_input = 32
n_predictions = 128
# number of units in RNN cell
n_hidden = 512
minibatch_size = 12

# Target log path
SESSION_NAME = "{}-Layers_{}-mem_{}-units".format(4, learning_rate, n_input, n_hidden)
logs_path = "/logs/training/{}".format(SESSION_NAME)

training_folder = './cello_text'
save_loc = './resources/models/model.ckpt'
loader = DataLoader(n_input, minibatch_size, training_folder, None)
loader.loadData(True)
vocab_size = len(loader.dictionary)

# tf Graph input
x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, vocab_size])

# RNN output node weights and biases
weights = {
	'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
}
biases = {
	'out': tf.Variable(tf.random_normal([vocab_size]))
}

def RNN(x, weights, biases):

	def make_cell(n_hidden):
		cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
		# TODO if is_training and keep_prob < 1:
		cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.8)
		return cell

	# reshape to [1, n_input]
	x = tf.reshape(x, [-1, n_input])

	# Generate a n_input-element sequence of inputs
	# (eg. [had] [a] [general] -> [20] [6] [33])
	x = tf.split(x, n_input, 1)

	rnn_cell = rnn.MultiRNNCell([make_cell(n_hidden), make_cell(n_hidden), make_cell(n_hidden), make_cell(n_hidden)])

	# generate prediction
	outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

	# there are n_input outputs but
	# we only want the last output
	return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
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
	tf.summary.histogram("weights", weights['out'])
	tf.summary.histogram("biases", biases['out'])
	summary = tf.summary.merge_all()

# Launch the graph
with tf.Session() as session:
	session.run(init)

	step = iteration.eval()
	acc_total = 0
	loss_total = 0

	training_writer = tf.summary.FileWriter('./logs/{}/training'.format(SESSION_NAME), session.graph)

	while step < training_iters or training_iters < 0:
		try:
			minibatch, labels, usedOffset = loader.getNextMinibatch()

			symbols_in_keys = minibatch
			symbols_out_onehot = labels

			_, acc, loss, training_summary = session.run([optimizer, accuracy, cost, summary], feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
			loss_total += loss
			acc_total += acc
			training_writer.add_summary(training_summary, step)

			if (step+1) % display_step == 0:
				onehot_pred = session.run(pred, feed_dict={x: symbols_in_keys, y: symbols_out_onehot})

				print("[{}-trainingset] Iter= ".format(str(datetime.now())) + str(step + 1) + ", Average Loss= " + \
					  "{:.6f}".format(loss_total / display_step) + ", Average Accuracy= " + \
					  "{:.2f}%".format(100 * acc_total / display_step))

				session.run(iteration.assign(step))
				saver.save(session, save_loc)
				acc_total = 0
				loss_total = 0

				symbols_in = [loader.reverse_dictionary[i[0]] for i in symbols_in_keys[0]]
				symbols_out = loader.reverse_dictionary[int(tf.argmax(symbols_out_onehot[0], 0).eval())]
				symbols_out_pred = loader.reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval()[0])]

				print("%s - [%s] vs [%s]" % (symbols_in, symbols_out, symbols_out_pred))

			step += 1
		except KeyboardInterrupt:
			print("Exiting and saving model")
			session.run(iteration.assign(step))
			break

		saver.save(session, save_loc)

