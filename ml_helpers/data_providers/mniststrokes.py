import os
import tempfile

import numpy as np
from tqdm import tqdm


from .downloader import download_data_url
from .base_provider import DataSet, DataProvider


class MNISTStrokesDataSet(DataSet):
    def __init__(self, points, inputs, targets, labels, shuffle=True):
        self.points = np.array(points)
        self.inputs = np.array(inputs)
        self.targets = np.array(targets)
        self.labels = np.array(labels)
        self.shuffle = shuffle

        self.start_new_epoch()

    def start_new_epoch(self):
        self._batch_counter = 0
        if self.shuffle:
            arrays = self.points, self.inputs, self.targets, self.labels
            output = self.shuffle_N_arrays(arrays)
            self.points, self.inputs, self.targets, self.labels = output

    def shuffle_N_arrays(self, arrays):
        """Shuffle N numpy arrays with same indexes
        Args:
            arrays: list of numpy arrays
        Return:
            shuffled_arrays: list of numpy arrays
        """
        rand_indexes = np.random.permutation(arrays[0].shape[0])
        shuffled_arrays = []
        for array in arrays:
            shuffled_arrays.append(array[rand_indexes])
        return shuffled_arrays

    @property
    def num_examples(self):
        return self.labels.shape[0]

    def next_batch(self, batch_size):
        """Retrun new batch of required size
        Returns:
            points_slice
            inputs_slice
            targets_slice
            labels_slice
        """
        start = self._batch_counter * batch_size
        end = (self._batch_counter + 1) * batch_size
        self._batch_counter += 1
        points_slice = self.points[start: end]
        inputs_slice = self.inputs[start: end]
        targets_slice = self.targets[start: end]
        labels_slice = self.labels[start: end]
        if points_slice.shape[0] != batch_size:
            self.start_new_epoch()
            return self.next_batch(batch_size)
        else:
            return points_slice, inputs_slice, targets_slice, labels_slice


class MNISTStrokesDataProvider(DataProvider):
    data_url = (
        'https://github.com/edwin-de-jong/'
        'mnist-digits-stroke-sequence-data/raw/master/sequences.tar.gz')
    n_classes = 10

    def __init__(self, one_hot=True, save_path=None, validation_split=None,
                 shuffle=True, verbose=True):
        """
        Args:
            one_hot: `bool`, return lasels one hot encoded
            save_path: `str`, where downloaded data should be store.
                By default TMP directory will bbe used.
            validation_split: `float` or None
                float: chunk of `train set` will be marked as `validation set`.
            shuffle: `bool`, should shuffle data or not
            verbose: `bool`, should report some pregress
        """
        self._save_path = None
        download_data_url(self.data_url, self.save_path)
        sub_save_path = os.path.join(self.save_path, 'sequences')
        for dataset in ['test', 'train']:
            labels_file = os.path.join(sub_save_path, '%slabels.txt' % dataset)
            with open(labels_file, 'r') as f:
                labels = f.readlines()
            # labels, one_hot=False
            labels = [int(l) for l in labels]
            dataset_points = []
            dataset_inputs = []
            dataset_targets = []
            dataset_labels = np.array(labels)
            if one_hot:
                dataset_labels = self.labels_to_one_hot(dataset_labels)
            if verbose:
                description = 'Read %s dataset from text files' % dataset
                labels_range = enumerate(tqdm(labels, desc=description))
            else:
                labels_range = enumerate(labels)
            for label_idx, label in labels_range:
                part_f_name = '%simg-%d' % (dataset, label_idx)
                points_file = os.path.join(
                    sub_save_path, '%s-points.txt' % part_f_name)
                inputs_data_file = os.path.join(
                    sub_save_path, '%s-inputdata.txt' % part_f_name)
                targets_data_file = os.path.join(
                    sub_save_path, '%s-targetdata.txt' % part_f_name)

                label_points = []
                with open(points_file, 'r') as f:
                    for line in f.readlines():
                        try:
                            point = [int(i.strip()) for i in line.split(',')]
                            label_points.append(point)
                        except ValueError:
                            continue
                label_points = np.array(label_points)
                dataset_points.append(label_points)

                inputs = np.loadtxt(inputs_data_file, dtype='int')
                dataset_inputs.append(inputs)

                targets = np.loadtxt(targets_data_file, dtype='int')
                dataset_targets.append(targets)

            if dataset == 'test':
                self.test = MNISTStrokesDataSet(
                    dataset_points,
                    dataset_inputs,
                    dataset_targets,
                    dataset_labels,
                    shuffle=True)

            if dataset == 'train':
                if validation_split is not None:
                    split_idx = int(
                        dataset_labels.shape[0] * (1 - validation_split))
                    self.validation = MNISTStrokesDataSet(
                        dataset_points[split_idx:],
                        dataset_inputs[split_idx:],
                        dataset_targets[split_idx:],
                        dataset_labels[split_idx:],
                        shuffle=shuffle)
                    # crop datasets for train
                    dataset_points = dataset_points[:split_idx]
                    dataset_inputs = dataset_inputs[:split_idx]
                    dataset_targets = dataset_targets[:split_idx]
                    dataset_labels = dataset_labels[:split_idx]

                self.train = MNISTStrokesDataSet(
                    dataset_points,
                    dataset_inputs,
                    dataset_targets,
                    dataset_labels,
                    shuffle=shuffle)

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = os.path.join(
                tempfile.gettempdir(), 'mnist_stokes')
        return self._save_path

    @property
    def data_shape(self):
        return None


if __name__ == '__main__':
    provider = MNISTStrokesDataProvider(validation_split=0.1)
