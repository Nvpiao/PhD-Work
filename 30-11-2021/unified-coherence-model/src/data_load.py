import os
import random

from src.utils import load_file

'''
## Description of data-tokenized:
It is the dataset for local discrimination.
path/directory contains train/test/dev in different folder. 
Each of this filename contains pos and neg doc (3D list).  [pos, neg] -> [sent1, sent2, ..] -> [word1, wrod2, ..]
Data are tokenized.
'''


class BatchGeneratorLocal():
    def __init__(self, args, test=False):
        """
        - path: file directory path
        """
        self.file_type = args.file_type
        self.test = test
        if self.test == True:
            self.path = args.test_path
            self.batch_size = args.batch_size_test
            self.shuffle = False
        else:
            self.path = args.train_path
            self.batch_size = args.batch_size
            self.shuffle = args.shuffle

    def __iter__(self):
        items = os.listdir(self.path)
        if self.shuffle == True:
            random.shuffle(items)
        batch = []
        batch_fname = []
        batch_length = []
        for fname in items:
            loadpath = os.path.join(self.path, fname)
            batch_file = load_file(loadpath, self.file_type)
            batch.append(batch_file)
            batch_fname.append(fname)
            batch_length.append(len(batch_file[0]))
            if len(batch) == self.batch_size:
                yield batch, batch_length, batch_fname
                batch = []  # make it batch empty for the next iteration
                batch_fname = []
                batch_length = []


'''
###Description of Global Files:
In the file `filelist_path`, there are list of filenames. For a doc, there are more than 20 entries of its perm.
Each of this filename contains pos and neg doc (2d list).  [pos, neg] -> [sent1, sent2, ..] wherer sent1,.. are strings
'''


class BatchGeneratorGlobal():
    def __init__(self, args, test=False):
        """
        - path: file directory path
        - filelist_path: list of files in path directory
        """
        self.file_type = args.file_type
        self.test = test
        if self.test == True:
            self.path = args.test_path
            self.filelist_path = args.file_list_test
            self.batch_size = args.batch_size_test
            self.shuffle = False
        else:
            self.path = args.train_path
            self.filelist_path = args.file_list_train
            self.batch_size = args.batch_size_train
            self.shuffle = args.shuffle

    def __iter__(self):
        dir_paths = os.walk(self.path)

        batch = []
        batch_fname = []
        batch_length = []

        for dir_path in dir_paths:
            root_dir = dir_path[0]
            file_path_list = dir_path[2]
            if self.shuffle:
                random.shuffle(file_path_list)

            for fname in file_path_list:
                loadpath = os.path.join(self.path, fname)
                batch_file = load_file(loadpath, self.file_type)
                # if pos and neg are same file i.e. perm is same, skip it
                if batch_file[0] == batch_file[1]:
                    continue
                for z in range(len(batch_file)):  # z=0 -> pos_doc; z=1 -> neg_doc
                    batch_file[z] = [sentence.split()
                                     for sentence in batch_file[z]]
                batch.append(batch_file)
                batch_length.append(len(batch_file[0]))
                batch_fname.append(fname)
                if len(batch) == self.batch_size:
                    yield batch, batch_length, batch_fname
                    batch = []  # make it batch empty for the next iteration
                    batch_fname = []
                    batch_length = []


def create_batch_generators(args):
    if args.dataset == 'data-global':
        print("Reading Global Discrimination Dataset")
        batch_generator_train = BatchGeneratorGlobal(args)
        batch_generator_test = BatchGeneratorGlobal(args, test=True)
    else:
        print("Reading Local Discrimination Dataset")
        batch_generator_train = BatchGeneratorLocal(args)
        batch_generator_test = BatchGeneratorLocal(args, test=True)
    return batch_generator_train, batch_generator_test
