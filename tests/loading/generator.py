import numpy as np
import tensorflow as tf
import gc
from loading.disk_array import DiskArray


class DataGenerator(tf.keras.utils.Sequence):
    """
    Data generator for keras modeling
    """
    def __init__(self, X, Y, batch_size=512, shuffle=True):
        """
        If data length is not divisible by batch size, will keep out a random set of rows each round (who make up the
        remainder that don't fit into a full batch)


        :param X: input sequences. Can contain multiple arrays
        :param Y: output sequences
        :param batch_size:
        :param shuffle: Whether to shuffle data after each epoch
        """
        self.batch_size = batch_size
        if isinstance(X, list):
            self.num_X_sets = len(X)
            self.X_unified = [xset.copy() for xset in X]
        else:
            self.num_X_sets = 1
            self.X_unified = [X]
        self.Y_unified = Y.copy()
        self.X_split = None
        self.Y_split = None
        self.shuffle = shuffle
        self.data_is_split = False
        if isinstance(X, list):
            self.complete_len = len(X[0])
        else:
            self.complete_len = len(X)
        self.batch_len = int(np.floor(self.complete_len / self.batch_size))
        self.split_indexes = np.arange(0, self.complete_len, self.batch_size)[1:]
        self.on_epoch_end()

    def __len__(self):
        """
        :return: Number of batches per epoch
        """
        return self.batch_len

    def __getitem__(self, index):
        """
        Return one batch of data

        :param index: Index of batch to retrieve
        :return:
        """
        return self.__data_generation(index)

    def on_epoch_begin(self):
        """
        Split data into batches at the beginning of epoch

        :return:
        """
        if not self.data_is_split:
            self.X_split = [np.split(xset, self.split_indexes, axis=0) for xset in self.X_unified]
            self.Y_split = np.split(self.Y_unified, self.split_indexes, axis=0)
            self.X_unified = None
            self.Y_unified = None
            self.data_is_split = True

    def on_epoch_end(self):
        """
        Shuffle dataset at the end of an epoch

        :return:
        """
        if self.data_is_split:
            X_info = [[list(xset[0].shape), xset[0].dtype] for xset in self.X_split]
            for i in range(len(X_info)):
                X_info[i][0][0] = self.complete_len
            self.X_unified = [np.empty(shape, dtype=dtype) for shape, dtype in X_info]
            for i in range(self.num_X_sets):
                self.X_unified[i][:] = np.nan
            for xset_idx, start_index in zip(range(len(self.X_split[0])), range(0, self.complete_len, self.batch_size)):
                for i in range(self.num_X_sets):
                    self.X_unified[i][start_index:start_index+self.batch_size] = self.X_split[i][xset_idx]
                    self.X_split[i][xset_idx] = None



            self.Y_unified = np.concatenate(self.Y_split, axis=0)
            self.Y_split = None
            self.X_split = None
            self.data_is_split = False
        self.indexes = np.arange(self.complete_len)
        if self.shuffle:
            np.random.shuffle(self.indexes)

        self.X_unified = [xset[self.indexes] for xset in self.X_unified]
        self.Y_unified = self.Y_unified[self.indexes]


        self.X_split = [
            np.split(xset, self.split_indexes, axis=0) for xset in self.X_unified
        ]

        self.Y_split = np.split(self.Y_unified, self.split_indexes, axis=0)
        self.X_unified = None
        self.Y_unified = None
        self.data_is_split = True
        gc.collect()


    def __data_generation(self, index):
        """
        Retrieve a batch by index

        :param index: Index of batch
        :return:
        """
        output = self.Y_split[index]
        if self.num_X_sets == 1:
            input = self.X_split[0][index]
        else:
            input = {f'input_{i+1}':xset[index] for i, xset in enumerate(self.X_split)}

        return input, output