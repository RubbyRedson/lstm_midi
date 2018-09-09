import os.path
import pickle
import glob
import tensorflow as tf
import sys


class DataLoader:

	def __init__(self, n_input, training_folder, test_folder):
		self.n_input = n_input
		self.vocab_size = 0
		self.training_folder = training_folder
		self.test_folder = test_folder
		self.files = []
		self.dictionary = {}
		self.reverse_dictionary = {}
		self.nr_train_examples = 0
		self.nr_test_examples = 0

		if self.training_folder[-1] != '/':
			self.training_folder += '/'

		if self.test_folder[-1] != '/':
			self.test_folder += '/'

	def pickle_it(self, variable, name):
		pickle.dump(variable, open( name, "wb"))

	def load_pickle(self, name):
		return pickle.load(open( name, "rb"))

	def pickle_exists(self, name):
		return os.path.isfile(name)

	def build_dataset(self):
		dictionary = {}
		counter = 0
		for k in self.dictionary:
			dictionary[k] = counter
			counter += 1

		reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
		return dictionary, reverse_dictionary

	def cache_batches(self, prefix, input_folder):
		print("Creating cached batches for the training set")

		nr_of_examples = 0
		# open the TFRecords file
		val_filename = prefix + '.tfrecords'  # address to save the TFRecords file
		writer = tf.python_io.TFRecordWriter(val_filename)
		file_counter = 0
		for filename in glob.iglob(input_folder + '**/*.txt', recursive=True):

			data_row = self.read_data(filename)

			for data in data_row:
				batch_session = []

				for z in range(len(data) - 1):
					batch_session.append(self.dictionary[data[z]])

				label = [self.dictionary[data[-1]]]

				feature = {
					prefix + '/feature': tf.train.Feature(float_list=tf.train.FloatList(value=batch_session)),
					prefix + '/label': tf.train.Feature(int64_list=tf.train.Int64List(value=label))
				}

				example = tf.train.Example(features=tf.train.Features(feature=feature))
				writer.write(example.SerializeToString())
				nr_of_examples += 1

			file_counter += 1
			print("[{}/{} {:.2f}%] => {}".format(file_counter, len(self.files), (file_counter / len(self.files)) * 100, filename))

		writer.close()
		sys.stdout.flush()
		return nr_of_examples

	def read_data(self, fname):
		with open(fname) as f:
			lines = f.readlines()

		size_including_label = self.n_input + 1

		batched = []
		batch = []
		for i in range(len(lines)):
			for j in range(len(lines[i])):
				batch.append(lines[i][j])
				if len(batch) == size_including_label:
					batched.append(batch)
					batch = []

		# If not match up, pad with whitespace
		if len(batch) > 0:
			for i in range(size_including_label - len(batch)):
				batch.append(" ")
				batched.append(batch)

		return batched

	def read_folder(self, path):
		for filename in glob.iglob(path, recursive=True):

			self.files.append(filename)
			data = self.read_data(filename)

			for row in data:
				for item in row:
					self.dictionary[item] = 1
			print("[{}] => {}".format(len(self.dictionary), filename))

	def load_data(self, use_pickle=True):

		# Load the dictionaries from cache if exists and any of the data sets did not change
		if use_pickle \
				and self.pickle_exists("dictionary.p") \
				and self.pickle_exists("reverse_dictionary.p") \
				and self.pickle_exists("nr_train_examples.p") \
				and self.pickle_exists("nr_test_examples.p"):

			self.dictionary = self.load_pickle("dictionary.p")
			self.reverse_dictionary = self.load_pickle("reverse_dictionary.p")
			self.nr_train_examples = self.load_pickle("nr_train_examples.p")
			self.nr_test_examples = self.load_pickle("nr_test_examples.p")

		else:
			# Do a one pass and build the dictionary
			self.read_folder(self.training_folder + '**/*.txt')
			self.read_folder(self.test_folder + '**/*.txt')

			dictionary, reverse_dictionary = self.build_dataset()

			if use_pickle:
				self.pickle_it(dictionary, "dictionary.p")
				self.pickle_it(reverse_dictionary, "reverse_dictionary.p")

			self.dictionary = dictionary
			self.reverse_dictionary = reverse_dictionary

		self.vocab_size = len(self.dictionary)

		if not os.path.isfile("train.tfrecords"):
			self.nr_train_examples = self.cache_batches("train", self.training_folder)
			self.pickle_it(self.nr_train_examples, "nr_train_examples.p")

		if not os.path.isfile("test.tfrecords"):
			self.nr_test_examples = self.cache_batches("test", self.test_folder)
			self.pickle_it(self.nr_test_examples, "nr_test_examples.p")
