import numpy as np
from src.reader import ImageIO


class Fourier:
    def __init__(self, array_2d):
        assert len(array_2d.shape) == 2
        self.space_domain = array_2d
        self.frequency_domain = self.fourier_transform(self.space_domain)

        # cache
        self.cache_filtered_frequency = self.frequency_domain
        self.cache_space_domain = None

    def apply_frequency_filter(self, func):
        """
        Apply a filter to the frequency domain.
        :param func: function
        """
        if not func:
            self.cache_filtered_frequency = func(self.cache_filtered_frequency)
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

    def get_filter_space_domain(self, real=True):
        """
        Return 2D array with range (0, 1) or complex
        """
        if not self.cache_space_domain:
            self.cache_space_domain = reverse_fourier_domain(self.cache_filtered_frequency)

        if real:
            return np.real(self.cache_space_domain)
        else:
            return self.cache_space_domain

    @staticmethod
    def fourier_transform(time_domain_2d):
        # n - width of image, m - height of image

        fre_result = np.zeros_like(time_domain_2d, dtype=complex)

        m, n = time_domain_2d.shape
        for _m in range(0, m):
            row_vector = np.exp(-2.0 * np.pi * 1j * np.arange(0, m, 1.0) * _m / m)
            for _n in range(0, n):
                col_vector = np.exp(-2.0 * np.pi * 1j * np.arange(0, n, 1.0) * _n / n)

                fre_result[_m][_n] = np.dot(np.dot(row_vector, time_domain_2d), col_vector)
        return fre_result

    @staticmethod
    def reverse_fourier_transform(fre_domain_2d, size=None):
        if not size:
            m, n = fre_domain_2d.shape
        else:
            m, n = size

        fre_m, fre_n = fre_domain_2d.shape

        space_domain = np.zeros((m, n)).astype(complex)
        for _m in range(0, m):
            row_vector = np.exp(2 * np.pi * 1j * np.arange(0, fre_m, 1.0) * _m / m)
            for _n in range(0, n):
                col_vector = np.exp(2 * np.pi * 1j * np.arange(0, fre_n, 1.0) * _n / n)
                space_domain[_m][_n] = np.dot(np.dot(row_vector, fre_domain_2d), col_vector)

        return space_domain / (fre_m * fre_n)


if __name__ == '__main__':
    io = ImageIO('../image/lena.jpg')
    gray_array = io.get_gray()
    fourier = Fourier(gray_array)

    raw_space_domain = fourier.get_raw_space_domain()
    raw_fre_domain = fourier.get_raw_frequency_domain()

    frequency_visual = ImageIO.get_visual_frequency_domain(fourier.frequency_domain, center=True)
    enlarge_space_domain = Fourier.reverse_fourier_transform(raw_fre_domain, size=(200, 200))

    ImageIO.imshows([frequency_visual, raw_space_domain, np.real(enlarge_space_domain)])
