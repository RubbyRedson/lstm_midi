import os.path
import pickle
import random
import numpy as np
import glob

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
		if(self.training_folder[-1] != '/'):
			self.training_folder += '/'

	def isOutOfIndex(self):
		return self.offset > (len(self.training_data) - self.batch_size)

	def testSetIsOutOfIndex(self):
		return self.testOffset > (len(self.test_data) - self.batch_size)

	def getNextMinibatch(self):

		self.offset += self.batch_size

		if self.isOutOfIndex():
			self.offset = 0
			self.epoch += 1
			print("EPOCH => {}".format(self.epoch))

		minibatchItem = self.batches[self.offset: self.offset + self.batch_size].tolist()

		res = np.empty([self.batch_size, self.vocab_size])
		symbols_out_onehot = np.zeros([self.batch_size, self.vocab_size], dtype=float)

		for i in range(self.batch_size):
			symbols_out_onehot[i][minibatchItem[i][-1][0]] = 1.0
			minibatchItem[i] = minibatchItem[i][:-1]
			res[i] = np.reshape(symbols_out_onehot[i], [1, -1])[0]

		return minibatchItem, res, self.offset

	def getNextTestMiniBatch(self):
		self.testOffset += self.batch_size

		if self.testSetIsOutOfIndex():
			self.testOffset = 0
			self.testEpoch += 1
			print("TEST_EPOCH => {}".format(self.testEpoch))

		minibatchItem = self.test_batches[self.testOffset: self.testOffset + self.batch_size].tolist()

		res = np.empty([self.batch_size, self.vocab_size])
		symbols_out_onehot = np.zeros([self.batch_size, self.vocab_size], dtype=float)

		for i in range(self.batch_size):
			a = minibatchItem[i][-1]
			msg = self.reverse_dictionary[minibatchItem[i][-1][0]]
			symbols_out_onehot[i][minibatchItem[i][-1][0]] = 1.0
			minibatchItem[i] = minibatchItem[i][:-1]
			res[i] = np.reshape(symbols_out_onehot[i], [1, -1])[0]

		return minibatchItem, res, self.testOffset


	def pickleIt(self, variable, name):
		pickle.dump(variable, open( name, "wb"))

	def loadPickle(self, name):
		return pickle.load(open( name, "rb"))

	def pickleExists(self, name):
		return os.path.isfile(name)

	def build_dataset(self, trainingset, testingset):
		import collections
		import itertools

		flattenedTrainingset = list(itertools.chain.from_iterable(trainingset))

		if testingset is not None:
			flattenedTestingset = list(itertools.chain.from_iterable(testingset))
			flattened = flattenedTrainingset + flattenedTestingset
		else:
			flattened = flattenedTrainingset

		count = collections.Counter(flattened).most_common()
		dictionary = dict()
		for word, _ in count:
			dictionary[word] = len(dictionary)

		reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
		return dictionary, reverse_dictionary

	def cacheBatches(self):
		print("Creating cached batches for the training set")
		batches = []
		for chunk in self.training_data:
			batchSession = []
			for item in chunk:
				batchSession.append([self.dictionary[item]])
			batches.append(batchSession)

		self.batches = np.array(batches)
		np.random.shuffle(self.batches)

	def read_data(self, fname):
		with open(fname) as f:
			lines = f.readlines()

		content = [line.strip() for line in lines]

		size_including_label = self.n_input +1

		tmp = []
		for i in range(len(content)):
			pts = content[i].split(" ")
			if len(pts) == size_including_label:
				tmp.append(pts)
			elif len(pts) > size_including_label:
				for j in range(len(pts) - size_including_label):
					tmp.append(pts[j:j+size_including_label])
		content = tmp

		return content

	def filter(self, filename):
		return filename.endswith("_all.txt")

	def read_folder(self, path):
		content = []
		for filename in glob.iglob(path, recursive=True):
			if self.filter(filename):
				data = self.read_data(filename)
				if len(data) > 0:
					content += data

		return content

	def loadData(self, usePickle=True):

		# The dicts needs to be rebuilt if any of the data sets changed
		needToRebuildDict = False

		# Load training data from cahce if exists
		if usePickle and self.pickleExists("training_data.p"):
			self.training_data = self.loadPickle("training_data.p")
			print("Loaded training data from pickle...")
		else:
			needToRebuildDict = True
			self.training_data = self.read_folder(self.training_folder + '**/*.txt')
			if usePickle:
				self.pickleIt(self.training_data, "training_data.p")
				print("Loaded training data and pickled it...")

		# Load test data from cache if exists
		if usePickle and self.pickleExists("test_data.p"):
			self.test_data = self.loadPickle("test_data.p")
			print("Loaded test data from pickle...")
		else:
			needToRebuildDict = True
			if self.test_data is not None:
				self.test_data = self.read_folder(self.test_folder + '**/*.txt')

				if usePickle:
					self.pickleIt(self.test_data, "test_data.p")
					print("Loaded training data and pickled it...")

		# Load the dictionaries from cache if exists and any of the data sets did not change
		if(usePickle and self.pickleExists("dictionary.p") and self.pickleExists("reverse_dictionary.p")) and not needToRebuildDict:
			self.dictionary = self.loadPickle("dictionary.p")
			self.reverse_dictionary = self.loadPickle("reverse_dictionary.p")
		else:
			dictionary, reverse_dictionary = self.build_dataset(self.training_data, self.test_data)
			if usePickle:
				self.pickleIt(dictionary, "dictionary.p")
				self.pickleIt(reverse_dictionary, "reverse_dictionary.p")
			self.dictionary = dictionary
			self.reverse_dictionary = reverse_dictionary

		self.vocab_size = len(self.dictionary)
		self.cacheBatches()


