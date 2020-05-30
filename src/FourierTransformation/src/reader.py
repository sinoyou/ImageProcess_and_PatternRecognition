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
        R, G, B = [np.squeeze(x, axis=-1) for x in np.split(self.color_array, 3, axis=-1)]
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        U = -0.169 * R - 0.331 * G + 0.5 * B + 128
        V = 0.5 * R - 0.419 * G - 0.081 * B + 128
        yuv = [Y.astype(int), U.astype(int), V.astype(int)]
        return yuv

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
