import os
import numpy as np
from tensorflow.python.framework import dtypes, random_seed
from tensorflow.contrib.learn.python.learn.datasets import base


class DataSet(object):

    # Constructor
    def __init__(self,
                 images,
                 labels,
                 dtype=dtypes.float32,
                 seed=None):
        self.check_data(images, labels)
        seed1, seed2 = random_seed.get_seed(seed)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._total_batches = images.shape[0]

    # Check whether the number of images and number of labels are same
    def check_data(self, images, labels):
        assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))

    # Implementing getter methods
    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def total_batches(self):
        return self._total_batches

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """
        Takes in batch_size and returns a tuple of batch_size images
        and batch_size labels, for the next batch.
        """
        start = self._index_in_epoch
        # first epoch shuffle
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._total_batches)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]

        # next epoch
        if start + batch_size <= self._total_batches:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]

        # if the epoch is ending
        # append new images & labels after a full fresh shuffle, to the
        # left images & labels, and concatenate them to return batch_size
        # images and labels in a tuple
        else:
            self._epochs_completed += 1
            # store what is left of this epoch
            batches_left = self._total_batches - start
            images_left = self._images[start:self._total_batches]
            labels_left = self._labels[start:self._total_batches]
            # shuffle for new epoch
            if shuffle:
                perm = np.arange(self._total_batches)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
                # start next epoch
            start = 0
            self._index_in_epoch = batch_size - batches_left
            end = self._index_in_epoch
            images_new = self._images[start:end]
            labels_new = self._labels[start:end]
            return np.concatenate((images_left, images_new), axis=0), np.concatenate((labels_left, labels_new), axis=0)


def load_csv(fname, col_start=2, row_start=0, delimiter=","):
    data = np.genfromtxt(fname, delimiter=delimiter)
    # removing first 2 columns
    for _ in range(col_start):
        data = np.delete(data, (0), axis=1)
    # not removing any rows now, since row_start defaults to 0
    for _ in range(row_start):
        data = np.delete(data, (0), axis=0)
    return data


# stock data loading
def load_stock_data(path, moving_window=128, columns=5, train_test_ratio=4.0):

    # process a single file's data into usable arrays
    def process_data(data):

        # create an ndarray with shape (0, 128, 5) with no elements
        stock_set = np.zeros([0, moving_window, columns])

        # create an ndarray with shape (0, 2) with no elements
        label_set = np.zeros([0, 2])

        for idx in range(data.shape[0] - (moving_window + 5)):

            # Row bind the data of shape (1, 128, 5) to stock_set, for every index idx
            stock_set = np.concatenate((stock_set, np.expand_dims(data[range(idx, idx + (moving_window)), :], axis=0)),
                                       axis=0)

            # If the close price after 5 points is greater than the close price, then it's a bullish stock
            if data[idx + (moving_window + 5), 3] > data[idx + (moving_window), 3]:
                lbl = [[1.0, 0.0]]

            # Else, it's a bearish stock
            else:
                lbl = [[0.0, 1.0]]

            # Row bind the label of shape (1, 2) to label_set, for every index idx
            label_set = np.concatenate((label_set, lbl), axis=0)

        return stock_set, label_set

    # read a directory of data
    # do the same thing as above but for multiple stocks now
    stocks_set = np.zeros([0, moving_window, columns])
    labels_set = np.zeros([0, 2])
    for dir_item in os.listdir(path):
        dir_item_path = os.path.join(path, dir_item)
        if os.path.isfile(dir_item_path):
            print(dir_item_path)
            ss, ls = process_data(load_csv(dir_item_path))
            stocks_set = np.concatenate((stocks_set, ss), axis=0)
            labels_set = np.concatenate((labels_set, ls), axis=0)

    # shuffling the data
    perm = np.arange(labels_set.shape[0])
    np.random.shuffle(perm)
    stocks_set = stocks_set[perm]
    labels_set = labels_set[perm]

    # normalize the data
    # now each entry of a data point lies between 0 and 1
    stocks_set_ = np.zeros(stocks_set.shape)
    for i in range(len(stocks_set)):
        min = stocks_set[i].min(axis=0)
        max = stocks_set[i].max(axis=0)
        stocks_set_[i] = (stocks_set[i] - min) / (max - min)
    stocks_set = stocks_set_

    # selecting 1/5 for testing, and 4/5 for training
    train_test_idx = int((1.0 / (train_test_ratio + 1.0)) * labels_set.shape[0])
    train_stocks = stocks_set[train_test_idx:, :, :]
    train_labels = labels_set[train_test_idx:]
    test_stocks = stocks_set[:train_test_idx, :, :]
    test_labels = labels_set[:train_test_idx]

    train = DataSet(train_stocks, train_labels)
    test = DataSet(test_stocks, test_labels)

    return base.Datasets(train=train, validation=None, test=test)