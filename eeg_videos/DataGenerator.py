import numpy as np
import os

from keras.utils import Sequence
from threading import Lock


class NpyDataGenerator:
    extensions = {'npy'}

    def flow_from_dir(self,
                      directory,
                      batch_size=8,
                      shape=(120, 120),
                      validation_split=0.0,
                      preprocessing_function=None,
                      training_set=None,
                      classes=None,
                      dtype='float32',
                      shuffle=True):

        return Iterator(directory=directory,
                        batch_size=batch_size,
                        shape=shape,
                        extensions=self.extensions,
                        validation_split=validation_split,
                        preprocessing_function=preprocessing_function,
                        training_set=training_set,
                        classes=classes,
                        dtype=dtype,
                        shuffle=shuffle)


class Iterator(Sequence):
    def __init__(self,
                 directory,
                 batch_size,
                 shape,
                 extensions,
                 validation_split=0.0,
                 preprocessing_function=None,
                 training_set=None,
                 classes=None,
                 dtype='float32',
                 shuffle=True):
        self.directory = directory
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shape = shape
        self.dtype = dtype
        self.extensions = extensions
        self.validation_split = validation_split
        self.preprocessing_function = preprocessing_function
        self.classes = classes
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = Lock()
        self.index_array = None
        self.index_generator = self.__get_index()
        if training_set is None:
            self.split = None
        else:
            if self.validation_split < 0. or self.validation_split > 1.:
                raise ValueError('Validation split value should be between 0 and 1!')

            self.split = (0, self.validation_split) if not training_set else (self.validation_split, 1)

        self.n_samples = 0

        if not classes:
            classes = []
            for subdir in os.listdir(directory):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)

        self.n_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        self.filenames = []

        for subdir in classes:
            if self.split is not None:
                all_files = self.list_valid(directory, subdir, self.extensions)
                valid_files, actual_classes = all_files[self.split[0] * len(all_files):self.split[1] * len(all_files)]
            else:
                valid_files, actual_classes = self.list_valid(directory, subdir, self.extensions)

            if self.classes is None:
                self.classes = np.asarray(actual_classes, dtype='int32')
            else:
                self.classes = np.append(self.classes, actual_classes)

            self.filenames += valid_files

        self.n_samples = len(self.filenames)

        print('Found %d samples belonging to %d classes.' % (self.n_samples, self.n_classes))

    def __len__(self):
        """
        Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        if index > len(self):
            raise ValueError('Requested index %d exceeds the length of the current Sequence.' % index)

        self.total_batches_seen += 1

        if self.index_array is None:
            self.set_new_indices_array()

        indices_array = self.index_array[self.batch_size * index:self.batch_size * (index + 1)]

        return self.__get_batches(indices_array)

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            indices_array = next(self.index_generator)

        return self.__get_batches(indices_array)

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.set_new_indices_array()

    def __get_index(self):
        self.batch_index = 0

        while True:
            if self.batch_index == 0:
                self.set_new_indices_array()
            current_idx = (self.batch_index * self.batch_size) % self.n_samples

            if self.n_samples > current_idx + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0

            self.total_batches_seen += 1

            yield self.index_array[current_idx:current_idx + self.batch_size]

    def set_new_indices_array(self):
        self.index_array = np.arange(self.n_samples)

        if self.shuffle:
            self.index_array = np.random.permutation(self.n_samples)

    def __get_batches(self, indices_array):
        # build batch of data
        x = np.zeros((len(indices_array),) + self.shape, dtype=self.dtype)

        for i, j in enumerate(indices_array):
            data_x = np.load(self.filepaths[j])

            # if preprocessing function is specified, apply it
            if self.preprocessing_function is not None:
                data_x = self.preprocessing_function(data_x)

            x[i] = data_x

        # build batch of labels
        y = np.zeros((len(x), len(self.class_indices)), dtype=self.dtype)

        for i, n_observation in enumerate(indices_array):
            y[i, self.classes[n_observation]] = 1.

        return x, y

    def list_valid(self, directory, subdirectory, white_list_formats):
        valid_files = []
        classes = []

        for root, _, files in os.walk(os.path.join(directory, subdirectory)):
            for filename in files:
                for ext in white_list_formats:
                    if filename.lower().endswith('.' + ext):
                        valid_files.append(os.path.join(subdirectory, filename))
                        classes.append(self.class_indices[subdirectory])

        return valid_files, classes

    @property
    def filepaths(self):
        return [os.path.join(self.directory, fname) for fname in self.filenames]

    @property
    def labels(self):
        return self.classes
