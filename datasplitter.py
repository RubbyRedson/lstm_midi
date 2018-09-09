import os
import os.path
from shutil import copyfile
import glob


class Datasplitter():
	def __init__(self, input_dir, output_dir):
		self.input_dir = input_dir
		self.output_dir = output_dir

	def lazy_split(self):

		def lazy_create(path):
			if not os.path.isdir(path):
				os.makedirs(path)

		def filecount(path, ext):
			total = 0
			for root, dirs, files in os.walk(path):
				for file in files:
					if file.endswith(ext):
						total += 1
			return total

		lazy_create(self.output_dir + "/trainset")
		lazy_create(self.output_dir + "/testset")

		nr_of_files = filecount(self.input_dir, ".txt")
		split = 0.2
		split_on = int(nr_of_files / (nr_of_files * split))

		counter = 0
		for f in glob.iglob(self.input_dir + "/**/*.txt", recursive=True):

			split_folder = "/trainset"

			if(counter % split_on == 0):
				split_folder = "/testset"

			filename = f.split("/")
			filename = filename[-1]

			copyfile(f, self.output_dir + split_folder + "/" + filename)

			counter += 1

		files_in_testfolder = filecount(self.output_dir + "/testset", ".txt")
		files_in_trainfolder = filecount(self.output_dir + "/trainset", ".txt")

		print("{}/{}={} split as {:.2f}% testdata".format(files_in_testfolder, files_in_trainfolder, nr_of_files, (files_in_testfolder / nr_of_files) * 100))

if (__name__ == '__main__'):
	spliter = Datasplitter('/home/randomhash/Desktop/projects/midi_conversion/textfiles', './data')
	spliter.lazy_split()