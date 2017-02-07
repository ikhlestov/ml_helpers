import os

import matplotlib
import matplotlib.pyplot as plt


class BasePlotter:
    """
    Base class to provide most often used methods at plotters
    """

    def show(self, block=None):
        """
        Show previous built figure on display
        args:
            block: Bool, should script execution be blocked or not.
        """
        print("Matplotlib backend: %s" % matplotlib.rcParams['backend'])
        if block is not None:
            plt.show(block=block)
        else:
            plt.show()

    def savefig(self, path):
        """
        Save figure to required path
        args:
            path: str, path to file with required extension
        """
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        self.fig.savefig(path)
