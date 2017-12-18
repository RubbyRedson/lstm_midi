"""
TensorFlowOnSpark implementation of MIDI-trainer
"""


def test_fun(args, ctx):
	# Dependencies
	from tensorflowonspark import TFNode
	from datetime import datetime

	import getpass
	import math
	import numpy
	import os
	import random
	import signal
	import tensorflow as tf
	import time

	from tensorflow.contrib import rnn

	# Used for TensorBoard logdir
	from hops import tensorboard

	# Extract configuration
	worker_num = ctx.worker_num
	job_name = ctx.job_name
	task_index = ctx.task_index
	cluster_spec = ctx.cluster_spec
	num_workers = len(cluster_spec['worker'])

	# Get TF cluster/server instances
	cluster, server = TFNode.start_cluster_server(ctx, 1, args.rdma)

	# Parameters
	batch_size = 100
	display_iter = 1000
	training_iters = 50000

	learning_rate = 0.0001
	n_input = 3
	n_hidden = 512
	n_predictions = 32

	# Utility functions
	def elapsed(sec):
		if sec < 60:
			return str(sec) + " sec"
		elif sec < (60 * 60):
			return str(sec / 60) + " min"
		else:
			return str(sec / (60 * 60)) + " hr"

	def print_log(worker_num, arg):
		print("%d: " % worker_num)
		print(arg)

	def RNN(x, weights, biases, n_input, n_hidden):
		# Reshape to [1, n_input]
		x = tf.reshape(x, [-1, n_input])
		# Generate a n_input-element sequence of inputs
		# (eg. [had] [a] [general] -> [20] [6] [33])
		x = tf.split(x, n_input, 1)
		rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden), rnn.BasicLSTMCell(n_hidden)])

		# Generate prediction
		outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

		# There are n_input outputs but we only want the last output
		return tf.matmul(outputs[-1], weights['out']) + biases['out']

	def get_loss_fn(logits, labels):
		return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

	if job_name == "ps":
		server.join()
	elif job_name == "worker":
		# TODO What does this do?
		# Assigns ops to the local worker by default
		with tf.device(tf.train.replica_device_setter(
				worker_device="/job:worker/task:%d" % task_index,
				cluster=cluster)):

			# TODO Set up vocab_size by loading in dataset and parsing through it?
			dictionary = {}
			reverse_dictionary = {}
			vocab_size = 32

			# Placeholders or QueueRunner/Readers for input data
			num_epochs = 1 if args.mode == "inference" else None if args.epochs == 0 else args.epochs
			index = task_index if args.mode == "inference" else None
			workers = num_workers if args.mode == "inference" else None

			# RNN output node weights and biases
			hidden_weights = tf.Variable(tf.random_normal([n_hidden, vocab_size]), name="hidden_weights")
			hidden_biases = tf.Variable(tf.random_normal([vocab_size]), name="hidden_biases")
			weights = {'out': hidden_weights}
			biases = {'out': hidden_biases}

			# Graph input placeholders
			x = tf.placeholder("float", [None, n_input, 1])
			y = tf.placeholder("float", [None, vocab_size])

			# Set up TFOS
			global_step = tf.Variable(0)

			pred = RNN(x, weights, biases, n_input, n_hidden)
			cost = get_loss_fn(logits=pred, labels=y)
			# Note that the global_step is passed in to the optimizer's min. function
			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) \
				.minimize(loss=cost, global_step=global_step)

			# Model evaluation
			correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

			# TF summaries
			tf.summary.scalar("cost", cost)
			tf.summary.histogram("hidden_weights", hidden_weights)
			tf.summary.scalar("acc", accuracy)

			#  TODO XXX Below is copied directly from TFOS example
			saver = tf.train.Saver()
			summary_op = tf.summary.merge_all()
			init_op = tf.global_variables_initializer()

			# Create a "supervisor", which oversees the training process and stores model state into HDFS
			logdir = tensorboard.logdir()
			print("tensorflow model path: {0}".format(logdir))

			# Check if chief worker
			if job_name == "worker" and task_index == 0:
				summary_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())

			if args.mode == "train":
				sv = tf.train.Supervisor(is_chief=(task_index == 0),
				                         logdir=logdir,
				                         init_op=init_op,
				                         summary_op=None,
				                         summary_writer=None,
				                         saver=saver,
				                         global_step=global_step,
				                         stop_grace_secs=300,
				                         save_model_secs=10)
			else:
				sv = tf.train.Supervisor(is_chief=(task_index == 0),
				                         logdir=logdir,
				                         summary_op=None,
				                         saver=saver,
				                         global_step=global_step,
				                         stop_grace_secs=300,
				                         save_model_secs=0)
			# Configure output path on HDFS
			output_dir = TFNode.hdfs_path(ctx, args.output)
			output_file = tf.gfile.Open("{0}/part-{1:05d}".format(output_dir, worker_num), mode='w')

	# The supervisor takes care of session initialization, restoring from
	# a checkpoint, and closing when done or an error occurs.
	with sv.managed_session(server.target) as sess:
		print("{0} session ready".format(datetime.now().isoformat()))
		step = 0
		count = 0
		offset = random.randint(0, n_input + 1)
		end_offset = n_input + 1
		acc_total = 0
		loss_total = 0

		# TODO writer.add_graph(session.graph)? Might be taken care of by setup of summary_writer
		# TODO Set up args.steps

		# Loop until supervisor shuts down or max. iters have completed
		while not sv.should_stop() and step < args.steps:
			# TODO Determine what makes THIS asynch, and whether we need synch.
			# TODO A good resource may be https://stackoverflow.com/questions/41293576/distributed-tensorflow-good-example-for-synchronous-training-on-cpus
			# Run a training step asynchronously
			# See `tf.train.SyncReplicasOptimizer` for additional details on how to
			# perform *synchronous* training.

			# Using QueueRunner/Readers
			if args.mode == "train":
				# TODO Below is merely a copy-pasta of the local TF code, and will need refactoring
				if offset > (len(training_data) - end_offset):
					offset = random.randint(0, n_input + 1)

				symbols_in_keys = [[dictionary[str(training_data[i])]] for i in range(offset, offset + n_input)]
				symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])

				symbols_out_onehot = np.zeros([vocab_size], dtype=float)
				symbols_out_onehot[dictionary[str(training_data[offset + n_input])]] = 1.0
				symbols_out_onehot = np.reshape(symbols_out_onehot, [1, -1])

				# Run iteration and increment 'step'
				_, summary, acc, loss, onehot_pred, step = sess.run(
					[optimizer, summary_op, accuracy, cost, pred, global_step],
					feed_dict={x: symbols_in_keys, y: symbols_out_onehot})

				loss_total += loss
				acc_total += acc

				if ((step + 1) % display_iter) == 0:
					print("{0} step: {1} accuracy: {2}".format(
						datetime.now().isoformat(),
						step,
						sess.run(accuracy)))
					# TODO migrate over print fn from local TF code

				offset += (n_input + 1)

				if sv.is_chief:
					summary_writer.add_summary(summary, step)
			else:  # args.mode == "inference"
				# labels, pred, acc = sess.run([label, prediction, accuracy])
				# # print("label: {0}, pred: {1}".format(labels, pred))
				# print("acc: {0}".format(acc))
				# for i in range(len(labels)):
				# 	count += 1
				# 	output_file.write("{0} {1}\n".format(labels[i], pred[i]))
				# print("count: {0}".format(count))
				pass

		if args.mode == "inference":
			output_file.close()

		# Delay chief worker from shutting down supervisor during inference, since it can load model, start session,
		# run inference and request stop before the other workers even start/sync their sessions.
		if task_index == 0:
			time.sleep(60)

		# Ask for all the services to stop.
		print("{0} stopping supervisor".format(datetime.now().isoformat()))
		sv.stop()
