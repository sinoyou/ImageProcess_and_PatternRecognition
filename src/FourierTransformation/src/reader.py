import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from src.utils import mirror


class ImageIO:
    def __init__(self, path):
        color_image = Image.open(path)
        self.color_array = np.array(color_image, dtype=int)
        self.gray_array = np.array(color_image.convert('L'), dtype=int)

    def get_color(self):
        return self.color_array

    def get_gray(self, norm=True):
        if norm:
            return self.gray_array * 1.0 / 255
        else:
            return self.gray_array

    @staticmethod
    def get_visual_frequency_domain(frequency_domain, center=True, log=True):
        """
        Transform Frequency Domain Image to a visible format. [0 - 1]
        :param frequency_domain: Frequency Domain
        :param center: if True, reverse to highlight center
        """
        if log:
            f = np.log1p(np.abs(frequency_domain))
        else:
            f = np.abs(frequency_domain)

        f = (f - np.min(f)) / (np.max(f) - np.min(f))
        if center:
            f = mirror(f)

        return f

    @staticmethod
    def imshows(arrays, size=4):
        if isinstance(arrays[0], list):
            row = len(arrays)
            col = len(arrays[0])
        else:
            arrays = [arrays]
            row = len(arrays)
            col = len(arrays[0])
        fig, axes = plt.subplots(row, col, figsize=(size * col, size * row), squeeze=False)

        for _row in range(row):
            for _col in range(col):
                array = arrays[_row][_col]
                array = np.real(array)
                if np.max(array) <= 1:
                    array = (array * 255).astype(np.int)
                if len(array.shape) == 2:
                    array = np.stack([array, array, array], axis=2)
                axes[_row][_col].imshow(array)
        fig.show()


if __name__ == '__main__':
    i = ImageIO('landscape.png')
    gray = i.get_gray()
    i.imshows([gray, gray])
