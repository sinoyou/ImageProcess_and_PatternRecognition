import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from src.FourierTransformation.src.utils import mirror


class ImageIO:
    def __init__(self, path):
        color_image = Image.open(path)
        self.color_array = np.array(color_image, dtype=int)
        self.gray_array = np.array(color_image.convert('L'), dtype=int)

    def get_color(self):
        """
        Return (Y, U, V) as order.
        :return:
        """
        U = (self.color_array[..., 2] - self.gray_array) / 1.772
        V = (self.color_array[..., 0] - self.gray_array) / 1.402
        return self.gray_array, U.astype(np.int), V.astype(np.int)

    def get_gray(self, norm=True):
        if norm:
            return self.gray_array * 1.0 / 255
        else:
            return self.gray_array

    @staticmethod
    def yuv_to_rgb(yuv_list):
        Y, U, V = yuv_list[0], yuv_list[1], yuv_list[2]
        R = Y + 1.4075 * (V - 128)
        G = Y - 0.3455 * (U - 128) - 0.7169 * (V - 128)
        B = Y + 1.779 * (U - 128)
        return [R.astype(int), G.astype(int), B.astype(int)]

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
