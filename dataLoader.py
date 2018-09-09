import os.path
import pickle
import random
import numpy as np
import glob
import psutil
import tensorflow as tf
import sys

class DataLoader:

	training_data = None
	test_data = None
	dictionary = None
	reverse_dictionary = None
	offset = None
	testOffset = None
	n_input = None
	vocab_size = None
	training_folder = None
	test_folder = None
	batch_size = None
	batches = None
	test_batches = None
	epoch = None
	testEpoch = None
	training_count = None
	files = None

	def __init__(self, n_input, batch_size, training_folder, test_folder):
		self.offset = 0
		self.testOffset = 0
		self.n_input = n_input
		self.vocab_size = 0
		self.training_folder = training_folder
		self.test_folder = test_folder
		self.epoch = 0
		self.testEpoch = 0
		self.batch_size = batch_size
		self.files = []
		self.dictionary = {}
		if(self.training_folder[-1] != '/'):
			self.training_folder += '/'

	def isOutOfIndex(self):
		return self.offset > self.training_count - self.batch_size

	def testSetIsOutOfIndex(self):
		return self.testOffset > self.training_count - self.batch_size

	def increment(self):
		if self.isOutOfIndex():
			self.offset = 0
			self.epoch += 1
			print("EPOCH => {}".format(self.epoch))
		else:
			self.offset += self.batch_size

	def getNextMinibatch(self):

		if self.isOutOfIndex():
			self.offset = 0
			self.epoch += 1
			print("EPOCH => {}".format(self.epoch))

		features = self.batches[self.offset: self.offset + self.batch_size]
		labels = self.labels[self.offset: self.offset + self.batch_size]

		self.offset += self.batch_size

		return features, labels

	def getNextTestMiniBatch(self):
		self.testOffset += self.batch_size

		if self.testSetIsOutOfIndex():
			self.testOffset = 0
			self.testEpoch += 1
			print("TEST_EPOCH => {}".format(self.testEpoch))

		minibatchItem = self.test_batches[self.testOffset: self.testOffset + self.batch_size]
		newMiniBatches = []

		res = np.empty([self.batch_size, self.vocab_size])
		symbols_out_onehot = np.zeros([self.batch_size, self.vocab_size], dtype=float)

		for i in range(self.batch_size):
			newMiniBatches.append(minibatchItem[i].tolist())

			symbols_out_onehot[i][newMiniBatches[i][-1][0]] = 1.0
			newMiniBatches[i] = newMiniBatches[i][:-1]
			res[i] = np.reshape(symbols_out_onehot[i], [1, -1])[0]

		return newMiniBatches, res, self.testOffset


	def pickleIt(self, variable, name):
		pickle.dump(variable, open( name, "wb"))

	def loadPickle(self, name):
		return pickle.load(open( name, "rb"))

	def pickleExists(self, name):
		return os.path.isfile(name)

	def build_dataset(self, trainingset, testingset):
		dictionary = {}
		counter = 0
		for k in self.dictionary:
			dictionary[k] = counter
			counter += 1

		reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
		return dictionary, reverse_dictionary

	def cacheBatches(self):
		print("Creating cached batches for the training set")

		# open the TFRecords file
		val_filename = 'val.tfrecords'  # address to save the TFRecords file
		writer = tf.python_io.TFRecordWriter(val_filename)
		fileCounter = 0
		for filename in glob.iglob(self.training_folder  + '**/*.txt', recursive=True):

			if self.filter(filename):
				training_data_row = self.read_data(filename)
				
				for training_data in training_data_row:
					batchSession = []

					for z in range(len(training_data) - 1):
						batchSession.append(self.dictionary[training_data[z]])

					label = [self.dictionary[training_data[-1]]]

					feature = {
						'train/feature': tf.train.Feature(float_list=tf.train.FloatList(value=batchSession)),
						'train/label': tf.train.Feature(int64_list=tf.train.Int64List(value=label))
					}

					example = tf.train.Example(features=tf.train.Features(feature=feature))
					writer.write(example.SerializeToString())

				fileCounter += 1
				print("[{}/{} {:.2f}%] => {}".format(fileCounter, len(self.files), (fileCounter / len(self.files)) * 100, filename))
		writer.close()
		sys.stdout.flush()

	def read_data(self, fname):
		with open(fname) as f:
			lines = f.readlines()

		size_including_label = self.n_input +1

		batched = []
		batch = []
		for i in range(len(lines)):
			for j in range(len(lines[i])):
				batch.append(lines[i][j])
				if len(batch) == size_including_label:
					batched.append(batch)
					batch = []

		#If not match up, pad with whitespace
		if len(batch) > 0:
			for i in range(size_including_label - len(batch)):
				batch.append(" ")
				batched.append(batch)

		return batched

	def filter(self, filename):
		return True

	def read_folder(self, path):

		for filename in glob.iglob(path, recursive=True):
			if self.filter(filename):
				self.files.append(filename)

				data = self.read_data(filename)
				for row in data:
					for item in row:
						self.dictionary[item] = 1
				print("[{}] => {}".format(len(self.dictionary), filename))

		return self.dictionary

	def loadData(self, usePickle=True):
		needToRebuildDict = False

		# Load the dictionaries from cache if exists and any of the data sets did not change
		if(usePickle and self.pickleExists("dictionary.p") and self.pickleExists("reverse_dictionary.p")) and not needToRebuildDict:
			self.dictionary = self.loadPickle("dictionary.p")
			self.reverse_dictionary = self.loadPickle("reverse_dictionary.p")
		else:
			#Do a one pass and build the dictionary
			self.read_folder(self.training_folder + '**/*.txt')
			if self.test_folder is not None:
				self.read_folder(self.test_folder + '**/*.txt')
			dictionary, reverse_dictionary = self.build_dataset(self.training_data, self.test_data)

			if usePickle:
				self.pickleIt(dictionary, "dictionary.p")
				self.pickleIt(reverse_dictionary, "reverse_dictionary.p")
			self.dictionary = dictionary
			self.reverse_dictionary = reverse_dictionary

		self.vocab_size = len(self.dictionary)

		if not os.path.isfile("val.tfrecords"):
			self.cacheBatches()
