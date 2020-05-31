import numpy as np

# ref https://www.impulseadventure.com/photo/jpeg-quantization.html
canon_digital_fine_chr = np.array([
    [4, 4, 5, 9, 15, 26, 26, 26],
    [4, 4, 5, 10, 19, 26, 26, 26],
    [5, 5, 8, 9, 26, 26, 26, 26],
    [9, 10, 9, 13, 26, 26, 26, 26],
    [15, 19, 26, 26, 26, 26, 26, 26],
    [26, 26, 26, 26, 26, 26, 26, 26],
    [26, 26, 26, 26, 26, 26, 26, 26],
    [26, 26, 26, 26, 26, 26, 26, 26],
])

canon_digital_fine_lum = np.array([
    [1, 1, 1, 2, 3, 6, 8, 10],
    [1, 1, 2, 3, 4, 8, 9, 8],
    [2, 2, 2, 3, 6, 8, 10, 8],
    [2, 2, 3, 4, 7, 12, 11, 9],
    [3, 3, 8, 11, 10, 16, 15, 11],
    [3, 5, 8, 10, 12, 15, 16, 13],
    [7, 10, 11, 12, 15, 17, 17, 14],
    [14, 13, 13, 15, 15, 14, 14, 14],
])

slide_42_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 56],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

def quantize(dct, quant_matrix):
    assert dct.shape == quant_matrix.shape
    return np.divide(dct, quant_matrix).astype(int)


def back_quantize(feature_map: np.array, quant_matrix: np.array):
    assert feature_map.shape == quant_matrix.shape
    return np.multiply(feature_map, quant_matrix).astype(int)
