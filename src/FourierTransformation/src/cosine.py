import numpy as np
from src.FourierTransformation.src.reader import ImageIO
from src.FourierTransformation.src.frequency_domain_filter import *


class Cosine:
    def __init__(self, array_2d):
        assert len(array_2d.shape) == 2
        self.space_domain = array_2d
        self.frequency_domain = self.cosine_transform(self.space_domain)

        # cache
        self.cache_filtered_frequency = self.frequency_domain
        self.cache_space_domain = None

    def apply_frequency_filter(self, func, **kwargs):
        """
        Apply a filter to the frequency domain.
        :param func: function
        """
        if func:
            self.cache_filtered_frequency = func(self.frequency_domain, **kwargs)
            self.cache_space_domain = None

    def get_raw_frequency_domain(self):
        """
        Get original frequency domain from original image. 2D array of complex
        """
        return self.frequency_domain

    def get_filter_frequency_domain(self):
        """
        Get filtered frequency domain after applying filters on original frequency domain. 2D array of complex
        """
        return self.cache_filtered_frequency

    def get_raw_space_domain(self):
        """
        Return 2D array with range (0, 1)
        """
        return self.space_domain

    def get_filter_space_domain(self):
        """
        Return 2D array with range (0, 1) or complex
        """
        if not self.cache_space_domain:
            self.cache_space_domain = self.inverse_cosine_transform(self.cache_filtered_frequency)

        return self.cache_space_domain

    @staticmethod
    def cosine_transform(time_domain_2d):
        # n - width of image, m - height of image

        def get_c(loc):
            if loc == 0:
                return 1.0 / np.sqrt(2)
            else:
                return 1.0

        fre_result = np.zeros_like(time_domain_2d, dtype=float)
        m, n = time_domain_2d.shape
        for _m in range(0, m):
            row_vector = get_c(_m) * np.cos((2 * np.arange(0, m, 1.0) + 1) * _m * np.pi / (2 * m))
            for _n in range(0, n):
                col_vector = get_c(_n) * np.cos((2 * np.arange(0, n, 1.0) + 1) * _n * np.pi / (2 * n))

                fre_result[_m][_n] = np.dot(np.dot(row_vector, time_domain_2d), col_vector)

        fre_result = fre_result * 2 / np.sqrt(n * m)
        return fre_result

    @staticmethod
    def inverse_cosine_transform(fre_domain_2d, size=None):
        if not size:
            m, n = fre_domain_2d.shape
        else:
            m, n = size

        def get_c(loc):
            if loc == 0:
                return 1.0 / np.sqrt(2)
            else:
                return 1

        fre_m, fre_n = fre_domain_2d.shape

        space_domain = np.zeros((m, n)).astype(float)
        for _m in range(0, m):
            c_m = np.ones(fre_m)
            c_m[0] = get_c(0)
            row_vector = c_m * np.cos((2 * _m + 1) * np.arange(0, fre_m, 1.0) * np.pi / (2 * m))
            for _n in range(0, n):
                c_n = np.ones(fre_n)
                c_n[0] = get_c(0)
                col_vector = c_n * np.cos((2 * _n + 1) * np.arange(0, fre_n, 1.0) * np.pi / (2 * n))
                space_domain[_m][_n] = np.dot(np.dot(row_vector, fre_domain_2d), col_vector)

        return space_domain * 2 / np.sqrt(n * m)


if __name__ == '__main__':
    io = ImageIO('../image/lena.jpg')
    gray_array = io.get_gray()
    consine = Cosine(gray_array)

    # 原始内容
    raw_space = consine.get_raw_space_domain()
    raw_fre = consine.get_raw_frequency_domain()
    raw_fre_v = ImageIO.get_visual_frequency_domain(raw_fre, center=False)
    # ImageIO.imshows([raw_fre_v, raw_space])

    # 图像压缩
    consine.apply_frequency_filter(cosine_compress, c_size_r=100)
    c_fre = consine.get_filter_frequency_domain()
    c_fre_v = ImageIO.get_visual_frequency_domain(c_fre, center=False)
    c_space = consine.get_filter_space_domain()
    ImageIO.imshows([[raw_fre_v, raw_space], [c_fre_v, c_space]])
