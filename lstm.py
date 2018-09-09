from __future__ import print_function

import tensorflow as tf
from datetime import datetime
from dataLoader import DataLoader
from model import Model
import math

# Parameters
learning_rate = 0.001
training_iters = -1
nr_epochs = 100
display_step = 1000
n_input = 256

# number of units in RNN cell
n_hidden = 512
minibatch_size = 64
dropout = 0.50
n_layers = 4

# Target log path
SESSION_NAME = "{}-Layers_{}-LR_{}-mem_{}-units".format(n_layers, learning_rate, n_input, n_hidden)
logs_path = "/logs/training/{}".format(SESSION_NAME)
save_loc = './resources/models/model.ckpt'

loader = DataLoader(n_input, './data/trainset', './data/testset')
loader.load_data(True)

model = Model(n_hidden, n_input, n_layers, loader.vocab_size)


def parse_train(example_proto):
	features = {
		'train/feature': tf.FixedLenFeature((n_input), tf.float32),
		'train/label': tf.FixedLenFeature((), tf.int64)
	}

	parsed_features = tf.parse_single_example(example_proto, features)

	features = parsed_features["train/feature"]
	features = tf.reshape(features, [1, n_input])

	label = parsed_features["train/label"]
	label = tf.reshape(label, [1, 1])

	_y = tf.one_hot(label, loader.vocab_size, axis=1)
	_y = tf.reshape(_y, [loader.vocab_size])

	return features, _y

dataset = tf.data.TFRecordDataset("train.tfrecords")
dataset = dataset.map(parse_train, num_parallel_calls=4)
dataset = dataset.repeat().batch(minibatch_size).shuffle(buffer_size=10000).prefetch(1000)

iter = dataset.make_initializable_iterator()
x, y = iter.get_next()
init_op = iter.initializer

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
graph_epoch = tf.Variable(0, name="epoch")

# Initializing the variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()

# Create a variable scope for the logging
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
	# saver.restore(session, save_loc)
	step = iteration.eval()
	epoch = graph_epoch.eval()

	acc_total = 0
	loss_total = 0

	training_writer = tf.summary.FileWriter('./logs/{}/training'.format(SESSION_NAME), session.graph)
	training_iters = int(math.floor(loader.nr_train_examples / minibatch_size))

	while epoch < nr_epochs:
		session.run(init_op)

		while step < training_iters:
			try:
				if (step+1) % display_step == 0:
					_, acc, loss, _, _, training_summary = session.run([optimizer, accuracy, cost, iteration.assign(step), graph_epoch.assign(epoch), summary])

					loss_total += loss
					acc_total += acc

					print("[{}-trainingset] Iter= ".format(str(datetime.now())) + str(step + 1) + ", Average Loss= " +
											"{:.6f}".format(loss_total / display_step) + ", Average Accuracy= " +
											"{:.2f}%".format(100 * acc_total / display_step))

					training_writer.add_summary(training_summary, step)
					saver.save(session, save_loc)
					acc_total = 0
					loss_total = 0

				else:
					_, acc, loss = session.run([optimizer, accuracy, cost])

					loss_total += loss
					acc_total += acc

				step += 1
			except KeyboardInterrupt:
				print("Exiting and saving model")
				session.run(iteration.assign(step), graph_epoch.assign(epoch))
				break

		step = 0
		epoch += 1
		print("EPOCH {}/{}".format(epoch, nr_epochs))

	saver.save(session, save_loc)