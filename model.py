import tensorflow as tf
from tensorflow.contrib import rnn

class Model:
	n_hidden = None
	n_input = None
	vocab_size = None
	weights = None
	biases = None

	def __init__(self, n_hidden, n_input, n_layers, vocab_size):
		self.n_hidden = n_hidden
		self.n_input = n_input
		self.vocab_size = vocab_size
		self.n_layers = n_layers

		self.x = tf.placeholder("float", [None, self.n_input, 1], name="model.x")
		self.y = tf.placeholder("float", [None, self.vocab_size], name="model.y")

		self.weights = {
			'out': tf.Variable(tf.truncated_normal([self.n_hidden, self.vocab_size], stddev=0.1))
		}

		self.biases = {
			'out': tf.Variable(tf.constant(0.1, shape=[self.vocab_size]))
		}

		# Inverse of dropout
		self.pkeep = tf.placeholder(tf.float32)

	def RNN(self):
		def make_cell(n_hidden):
			cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
			# cell = rnn.BasicLSTMCell(n_hidden)
			# TODO if is_training and keep_prob < 1:
			cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.pkeep)
			return cell

		# reshape to [1, n_input]
		x = tf.reshape(self.x, [-1, self.n_input])

		# Generate a n_input-element sequence of inputs
		x = tf.split(x, self.n_input, 1)

		rnn_cell = rnn.MultiRNNCell([make_cell(self.n_hidden) for _ in range(self.n_layers)])

		# generate prediction
		outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

		# there are n_input outputs but
		# we only want the last output
		return tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']