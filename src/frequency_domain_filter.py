import numpy as np
from src.utils import mirror


def ideal_low_pass_filter(fre_domain, **kwargs):
    """
    理想低通滤波器
    允许指定的参数：
     - D_0(u, v)
    """
    if 'clip_d' not in kwargs.keys():
        print('Clip D set to 50 by default.')
        clip_d = 50
    else:
        clip_d = kwargs['clip_d']

    height, width = fre_domain.shape
    center = height // 2, width // 2
    mirror_fre_domain = mirror(fre_domain)
    mask = np.zeros_like(fre_domain)
    for h in range(height):
        for w in range(width):
            if (h - center[0]) ** 2 + (w - center[1]) ** 2 <= clip_d ** 2:
                mask[h][w] = 1.0
    result = mirror(np.multiply(mask, mirror_fre_domain))
    return result


def butterworth_low_pass_filter(fre_domain, **kwargs):
    """
    ButterWorth 低通滤波器
    允许指定的参数
     - 截断频率 D_0(u, v)
    """
    if 'clip_d' not in kwargs.keys():
        print('Clip D set to 50 by default.')
        clip_d = 50
    else:
        clip_d = kwargs['clip_d']

    height, width = fre_domain.shape
    center = height // 2, width // 2
    mirror_fre_domain = mirror(fre_domain)
    mask = np.zeros_like(fre_domain)
    for h in range(height):
        for w in range(width):
            d = (h - center[0]) ** 2 + (w - center[1]) ** 2
            mask[h][w] = 1.0 / (1.0 + d / (clip_d ** 2))
    result = mirror(np.multiply(mask, mirror_fre_domain))
    return result


def double_butterworth_filter(fre_domain, **kwargs):
    l_clip_d = kwargs['l_clip_d']
    h_clip_d = kwargs['h_clip_d']
    exp_rate = kwargs['exp_rate']

    height, width = fre_domain.shape
    center = height // 2, width // 2
    mirror_fre_domain = mirror(fre_domain)
    mask = np.zeros_like(fre_domain)
    for h in range(height):
        for w in range(width):
            d = np.sqrt((h - center[0]) ** 2 + (w - center[1]) ** 2)
            mask_l = 1.0 / (1.0 + (d / l_clip_d) ** exp_rate)
            mask_h = 1 - 1.0 / (1.0 + (d / h_clip_d) ** exp_rate)
            mask[h][w] = mask_l + mask_h
    result = mirror(np.multiply(mask, mirror_fre_domain))
    return result


def cosine_compress(fre_domain, **kwargs):
    """
    对DCT变换后的频域信息进行压缩，仅取左上角的内容，剩余内容为黑色
    """
    c_size_r = kwargs['c_size_r']
    _mask = np.zeros_like(fre_domain)
    m, n = fre_domain.shape
    for _m in range(m):
        for _n in range(n):
            if _m ** 2 + _n ** 2 <= c_size_r ** 2:
                _mask[_m][_n] = 1.0
    return fre_domain * _mask
