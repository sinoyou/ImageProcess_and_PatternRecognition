import numpy as np

high_quality = np.zeros((8, 8))


def quantize(dct, quant_matrix):
    assert dct.shape == quant_matrix.shape
    return np.divide(dct, quant_matrix).astype(np.int)


def back_quantize(feature_map: np.array, quant_matrix: np.array):
    assert feature_map.shape == quant_matrix.shape
    return np.multiply(feature_map, quant_matrix).astype(np.int)