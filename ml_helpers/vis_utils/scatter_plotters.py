from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from .n_classes_color_map import ColorMap
from .base import BasePlotter


class Plotter2D(BasePlotter):
    """
    Plot 2D scatter plot. If labels were provided classes will be dynamically
    colored.
    """

    def __init__(self, x_array, y_array, labels=None, alpha=1.0, grid=True,
                 legend_loc='upper right', title=''):
        """
        args:
            x_array: 1D numpy array or list
            y_array: 1D numpy array or list
            labels: 1D numpy array or list. default None.
        """
        assert np.array(x_array).shape[0] == np.array(y_array).shape[0]
        if labels is not None:
            assert np.array(x_array).shape[0] == np.array(labels).shape[0]

        self.fig = plt.figure(figsize=(12, 12))
        axe = self.fig.add_subplot(111)
        axe.grid(grid)
        if title:
            axe.set_title(title)

        if labels is None:
            axe.scatter(x_array, y_array, alpha=alpha)

        else:
            labels = np.array(labels)
            print(labels[:10])
            unique_labels = np.unique(labels)
            n_classes = len(unique_labels)
            color_map = ColorMap(n_classes)
            # TODO: check that mask work as expected
            print('-'*10)
            for class_idx, label in enumerate(unique_labels):
                mask = np.where(labels == label)[0]
                print("mask len", len(mask))
                axe.scatter(
                    x_array[mask],
                    y_array[mask],
                    c=color_map(class_idx),
                    label=str(label),
                    alpha=alpha)
            axe.legend(loc=legend_loc)


class Plotter3D(BasePlotter):
    """
    Plot 3D scatter plot. If labels were provided classes will be dynamically
    colored.
    """

    def __init__(self, x_array, y_array, z_array, labels=None, alpha=1.0,
                 grid=True, legend_loc='upper right', title=''):
        """
        args:
            x_array: 1D numpy array or list
            y_array: 1D numpy array or list
            z_array: 1D numpy array or list
            labels: 1D numpy array or list. default None.
        """
        assert np.array(x_array).shape[0] == np.array(y_array).shape[0]
        assert np.array(y_array).shape[0] == np.array(z_array).shape[0]
        if labels is not None:
            assert np.array(x_array).shape[0] == np.array(labels).shape[0]

        self.fig = plt.figure(figsize=(12, 12))
        axe = self.fig.add_subplot(111, projection='3d')
        axe.grid(grid)
        if title:
            axe.set_title(title)

        if labels is None:
            axe.scatter(x_array, y_array, z_array, alpha=alpha)

        else:
            labels = np.array(labels)
            unique_labels = np.unique(labels)
            n_classes = len(unique_labels)
            color_map = ColorMap(n_classes)
            for class_idx, label in enumerate(unique_labels):
                mask = np.where(labels == label)[0]
                axe.scatter(
                    x_array[mask],
                    y_array[mask],
                    z_array[mask],
                    c=color_map(class_idx),
                    label=str(label),
                    alpha=alpha)
            axe.legend(loc=legend_loc)


if __name__ == '__main__':
    x_array = np.random.rand(100)
    y_array = np.random.rand(100)
    labels = list('abcdefghqw' * 10)
    scatter_plot = Plotter2D(x_array, y_array)
    scatter_plot.show()
    scatter_plot = Plotter2D(x_array, y_array, labels)
    scatter_plot.show()

    x_array = np.random.rand(100)
    y_array = np.random.rand(100)
    z_array = np.random.rand(100)
    labels = list('abcdefghqw' * 10)
    scatter_plot = Plotter3D(x_array, y_array, z_array)
    scatter_plot.show()
    scatter_plot = Plotter3D(x_array, y_array, z_array, labels)
    scatter_plot.show()
