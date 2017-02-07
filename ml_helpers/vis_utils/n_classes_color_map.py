import matplotlib.cm as cmx
import matplotlib.colors as colors


class ColorMap:
    """Generate color map with N distinct colors

    example:
        # generate color map for 10 classes
        color_map = ColorMap(10)
        # get color for 2nd class(indexes 0th based)
        color_map(1)
    """

    def __init__(self, n_classes, cmap='hsv'):
        """
        args:
            n_classes: `int`, total quantity of classes
            cmap: `str`, optional color map. check this link
                http://matplotlib.org/examples/color/colormaps_reference.html
        """
        self.n_classes = n_classes - 1
        color_norm = colors.Normalize(vmin=0, vmax=n_classes)
        self.scalar_map = cmx.ScalarMappable(norm=color_norm, cmap=cmap)

    def __call__(self, class_idx):
        """
        args:
            class_idx: `int`, for what class color should be generated
        """
        if class_idx > self.n_classes:
            raise IndexError(
                "class idx {} out of range of available classes {}".format(
                    class_idx, self.n_classes))
        return self.scalar_map.to_rgba(class_idx)


if __name__ == '__main__':
    N = 10
    cmap = ColorMap(N)
    cmap(N - 1)
    try:
        cmap(N + 1)
    except Exception as e:
        assert isinstance(e, IndexError)
