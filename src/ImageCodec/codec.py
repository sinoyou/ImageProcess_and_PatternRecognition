import numpy as np
import matplotlib.pyplot as plt
from src.FourierTransformation.src.reader import ImageIO
from src.FourierTransformation.src.cosine import Cosine
from src.ImageCodec.quantization import quantize, back_quantize


class ImageCodec:
    def __init__(self, raw_path, quantization, gray=False):
        image_io = ImageIO(raw_path)
        self.quantization = quantization if isinstance(quantization, list) else [quantization]
        self.is_gray = gray
        self.raw_yuv = image_io.color_array()

        # property
        self.real_h = self.raw_yuv[0].shape[0]
        self.real_w = self.raw_yuv[0].shape[1]
        self.padding_h = (self.real_h // 8 + 1) * 8 if self.real_h % 8 != 0 else self.real_h
        self.padding_w = (self.real_w // 8 + 1) * 8 if self.real_w % 8 != 0 else self.real_w

        # preprocess
        self.padding_yuv = self.preprocess()

        # encode and quantization
        self.encode_list = self.encode()

        # decode
        self.decode_list = self.decode()

        # get compressed image
        self.compressed_yuv = self.get_compressed_image()

    def preprocess(self):
        """
        Padding image width and height not aligned with 8
        """
        padding_yuv = list()
        for x in self.raw_yuv:
            padding = np.zeros((self.padding_h, self.padding_w), dtype=np.int)
            padding[:self.real_h, :self.real_w] = x
            padding_yuv.append(padding)
        return padding_yuv

    def encode(self):
        """
        Run DCT Compress Algorithm. Return final and middle result.
        :return:
            space_block: original 8x8 image space signal blocks shaped like arrays. [-128, 128]
            dct_block: original 8x8 image frequency signal blocks shpaed like arrays.
            dct_quantize_block: int(dct_block / quantization_matrix)
        """
        result = []
        space_block = np.zeros((self.padding_h / 8, self.padding_w / 8, 8, 8), dtype=int)
        dct_block = np.zeros((self.padding_h / 8, self.padding_w / 8, 8, 8), dtype=float)
        dct_quantize_block = np.zeros((self.padding_h / 8, self.padding_w / 8), dtype=int)

        # loop for different channels in YUV
        for index, channel in enumerate(self.padding_yuv):
            # loop for height
            for ib in range(self.padding_h / 8):
                # loop for width
                for jb in range(self.padding_w / 8):
                    space_block[ib, jb] = channel[ib * 8: ib * 8 + 8, jb * 8: jb * 8 + 8] - 128
                    dct_block = Cosine.cosine_transform(space_block[ib, jb])
                    # support single quantization matrix or double quantization matrix.
                    if len(self.quantization) == 2 and index > 0:
                        dct_quantize_block = dct_block[ib, jb] / self.quantization[1]
                    else:
                        dct_quantize_block = dct_block[ib, jb] / self.quantization[0]

            result.append({'space_block': space_block,
                           'dct_block': dct_block,
                           'dct_quantize_block': dct_quantize_block})

        return result

    def decode(self):
        """
        Run DCT Decode Algorithm and return final & internal result.
        :return:
            compress_space_block: compressed 8x8 image space signal shaped in arrays. [-128, 128]
            compress_dct_block: compressed 8x8 image frequency signal shaped in arrays.
        """
        result = []
        compress_space_block = np.zeros((self.padding_h / 8, self.padding_w / 8, 8, 8), dtype=int)
        compress_dct_block = np.zeros((self.padding_h / 8, self.padding_w / 8, 8, 8), dtype=int)

        # loop for different channels in YUV
        for channel in self.encode_list:
            # loop for height
            for ib in range(self.padding_h / 8):
                # loop for width
                for jb in range(self.padding_w / 8):
                    dct_quantize_block = channel['dct_quantize_block']
                    compress_dct_block[ib, jb] = dct_quantize_block * self.quantization
                    compress_space_block[ib, jb] = Cosine.inverse_cosine_transform(compress_dct_block[ib, jb])
            result.append({'compress_space_block': compress_space_block,
                           'compress_dct_block': compress_dct_block})

        return result

    def get_compressed_image(self):
        """
        Revert Image singal of blocks into original [padding_height, padding_width] shape.
        :return: list of (Y, U, V) [0, 255]
        """
        result = []

        # loop for different channels in YUV
        for channel in self.decode_list:
            holder = np.zeros((self.padding_h, self.padding_w))
            for ib in range(self.padding_h / 8):
                for jb in range(self.padding_w / 8):
                    holder[ib * 8: ib * 8 + 8, jb * 8: jb * 8 + 8] = channel['compress_space_block'][ib, jb] + 128
            result.append(holder)

        return result

    def export_image(self, axes, compare=True):
        if self.is_gray:
            src = self.raw_yuv[0]
            dst = self.compressed_yuv[0]
        else:
            src = np.stack(ImageIO.yuv_to_rgb(self.raw_yuv), axis=2)
            dst = np.stack(ImageIO.yuv_to_rgb(self.compressed_yuv), axis=2)

        src = src[self.real_h, self.real_w]
        dst = dst[self.real_h, self.real_w]

        if compare:
            axes[0].imshow(src)
            axes[1].imshow(dst)
        else:
            axes.imshow(dst)

        return axes

    def storage_compress_evaluate(self):
        # 使用DC + AC 分流的Huffman Baseline 编码技术
        pass

    def quality_evaluate(self, metrics):
        pass
