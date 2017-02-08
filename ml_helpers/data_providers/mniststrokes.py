import os
import tempfile
import pickle

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

        self._start_new_epoch()

    def _start_new_epoch(self):
        self._batch_counter = 0
        if self.shuffle:
            arrays = self.points, self.inputs, self.targets, self.labels
            output = self._shuffle_N_arrays(arrays)
            self.points, self.inputs, self.targets, self.labels = output

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
            self._start_new_epoch()
            return self.next_batch(batch_size)
        else:
            return points_slice, inputs_slice, targets_slice, labels_slice


class MNISTStrokesDataProvider(DataProvider):
    data_url = (
        'https://github.com/edwin-de-jong/'
        'mnist-digits-stroke-sequence-data/raw/master/sequences.tar.gz')
    n_classes = 10

    def __init__(self, one_hot=True, save_path=None, validation_split=None,
                 shuffle=True, verbose=True, use_cache=True):
        """
        Args:
            one_hot: `bool`, return lasels one hot encoded
            save_path: `str`, where downloaded data should be store.
                By default TMP directory will bbe used.
            validation_split: `float` or None
                float: chunk of `train set` will be marked as `validation set`.
            shuffle: `bool`, should shuffle data or not
            verbose: `bool`, should report some pregress
            use_cache: `bool`, try to pickle/unpicke  processed data
        """
        self.use_cache = True
        self._save_path = None
        self._verbose = verbose
        self._one_hot = one_hot
        download_data_url(self.data_url, self.save_path, verbose=verbose)
        for dataset in ['test', 'train']:
            if self.use_cache:
                success, data = self._load_cached_data(dataset)
            else:
                success = False
            if not success:
                data = self._get_new_data(dataset)
            if self.use_cache and not success:
                self._save_cached_data(dataset, data)
            (dataset_points, dataset_inputs,
             dataset_targets, dataset_labels) = data

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

    def _get_new_data(self, dataset):
        sub_save_path = os.path.join(self.save_path, 'sequences')
        labels_file = os.path.join(sub_save_path, '%slabels.txt' % dataset)
        with open(labels_file, 'r') as f:
            labels = f.readlines()
        # labels, one_hot=False
        labels = [int(l) for l in labels]
        dataset_points = []
        dataset_inputs = []
        dataset_targets = []
        dataset_labels = np.array(labels)
        if self._one_hot:
            dataset_labels = self.labels_to_one_hot(dataset_labels)
        if self._verbose:
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
        return dataset_points, dataset_inputs, dataset_targets, dataset_labels

    def _get_pickle_filename(self, dataset):
        cache_dir = os.path.join(
            tempfile.gettempdir(), 'mnist_strokes', 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        pickle_filename = os.path.join(cache_dir, '%s_data.pkl' % dataset)
        return pickle_filename

    def _load_cached_data(self, dataset):
        pickle_filename = self._get_pickle_filename(dataset)
        try:
            with open(pickle_filename, 'rb') as f:
                data = pickle.load(f)
                labels = data[3]
                if self._one_hot and len(labels.shape) == 1:
                    labels = self.labels_to_one_hot(labels)
                    data = (data[0], data[1], data[2], labels)
                if not self._one_hot and len(labels.shape) == 2:
                    labels = self.labels_from_one_hot(labels)
                    data = (data[0], data[1], data[2], labels)
                success = True
        except Exception as e:
            if self._verbose:
                print("Failed to load cache with exception: %s" % str(e))
            success = False
            data = None
        return success, data

    def _save_cached_data(self, dataset, data_to_save):
        pickle_filename = self._get_pickle_filename(dataset)
        with open(pickle_filename, 'wb') as f:
            pickle.dump(data_to_save, f)

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = os.path.join(
                tempfile.gettempdir(), 'mnist_strokes')
        return self._save_path

    @property
    def data_shape(self):
        return None


if __name__ == '__main__':
    provider = MNISTStrokesDataProvider(validation_split=0.1, verbose=False)
