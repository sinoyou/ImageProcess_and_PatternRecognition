import numpy as np


def mirror(array):
    """
    Apply mirror transform to 2D array.
    The array will be equally split to 4 parts by vertically and horizontally.
    Then in each part, up-down and left-right flip will applied.
    Finally, 4 parts will be concat as original order.

    Application Area: frequency domain visualization, preprocess of frequency filter and etc.
    HINT: 此操作是可逆的，两次应用后可以恢复原始情况。
    :return:
    """
    f = array
    width_half = f.shape[1] // 2
    height_half = f.shape[0] // 2
    left_up = np.fliplr(np.flipud(f[0:height_half, 0:width_half]))
    left_down = np.fliplr(np.flipud(f[height_half:, 0:width_half]))
    right_up = np.fliplr(np.flipud(f[0:height_half, width_half:]))
    right_down = np.fliplr(np.flipud(f[height_half:, width_half:]))
    up = np.concatenate([left_up, right_up], axis=1)
    down = np.concatenate([left_down, right_down], axis=1)
    f = np.concatenate([up, down], axis=0)
    return f
