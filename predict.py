import pickle
import tensorflow as tf
import numpy as np
from model import Model
from random import randint

save_loc = './resources/models/model.ckpt'

def loadPickle(name):
	return pickle.load(open(name, "rb"))


dictionary = loadPickle("dictionary.p")
vocab_size = len(dictionary)

reverse_dictionary = loadPickle("reverse_dictionary.p")

# Parameters
n_input = 32
n_predictions = 512
n_layers = 4

# number of units in RNN cell
n_hidden = 512

model = Model(n_hidden, n_input, n_layers, vocab_size)
pred = model.RNN()

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as session:
	session.run(init)
	saver.restore(session, save_loc)
	print("Loaded session")

	# Plug in any initial words here
	words = []
	for i in range(n_input):
		randomIndex = randint(0, len(dictionary))
		words.append(reverse_dictionary[randomIndex])

	prediction = ""
	for word in words:
		prediction += word + " "

	symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]

	for i in range(n_predictions):
		keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
		onehot_pred = tf.nn.softmax(session.run(pred, feed_dict={model.x: keys, model.pkeep: 0.95}))
		onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())

		symbols_in_keys = symbols_in_keys[1:]
		symbols_in_keys.append(onehot_pred_index)

		prediction += reverse_dictionary[onehot_pred_index] + " "

	print(prediction)