import numpy as np


class DpcmRlcEncoder:
    def __init__(self, dct_compress_matrix):
        self.grid_h = dct_compress_matrix.shape[0]
        self.grid_w = dct_compress_matrix.shape[1]
        self.block_h = dct_compress_matrix.shape[2]
        self.blocK_w = dct_compress_matrix.shape[3]
        self.dct_compress_matrix = dct_compress_matrix.reshape(self.grid_w * self.grid_h, 8, 8)
        self.encoded_blocks = [[] for _ in range(self.grid_h * self.grid_w)]

        # 获取DPCM（差分脉冲调制）编码下的序列-list1，并返回上述序列经过VLI变长整数编码下的序列-list2。
        self.rel_dc, self.vli_rel_dc = self.dpcm()

        # 以RLC（游程编码）逐个对每个block进行编码形成序列-list1，并返回上述序列的单位元组中的部分值经过VLI编码后的序列-list2。
        self.ac_rlc = list()
        self.vli_ac_rlc = list()
        for h in range(self.grid_h):
            for w in range(self.grid_w):
                ac_rlc, vli_ac_rlc = self.rlc(h, w)
                self.ac_rlc.append(ac_rlc)
                self.vli_ac_rlc.append(vli_ac_rlc)

    def dpcm(self):
        abs_dc_value = [self.dct_compress_matrix[i, 0, 0] for i in range(self.grid_h * self.grid_w)]
        rel_dc_value = list()
        rel_dc_value.append(abs_dc_value[0])
        for i in range(1, self.grid_h * self.grid_w):
            rel_dc_value.append(abs_dc_value[i] - abs_dc_value[i - 1])

        # 将dpcm编码后的结果进一步地进行VLI编码，对于每个值，VLI将返回(row, index)，其中row是0～16的值，index = '' / 01字符串
        vli_rel_dc_value = [self.VLI_encode(x) for x in rel_dc_value]
        return rel_dc_value, vli_rel_dc_value

    def rlc(self, h, w):
        block_p = self.get_index(h, w)

        zig_zag_list = self.zig_zag(self.dct_compress_matrix[block_p], self.block_h, self.blocK_w)

        # encode zig_zag array with rlc rule
        rlc_list = []
        pre_zero_count = 0
        for p in range(len(zig_zag_list)):
            if zig_zag_list[p] != 0:
                rlc_list.append((pre_zero_count, zig_zag_list[p]))
                pre_zero_count = 0
            else:
                # 后续全部为0，以(0, 0) = <EOB> 修饰
                if sum([abs(x) for x in zig_zag_list[p:]]) == 0:
                    rlc_list.append((0, 0))
                    break
                else:
                    pre_zero_count += 1
                    # 中间的非零串以16个为一组
                    if pre_zero_count == 16:
                        rlc_list.append((15, 0))
                        pre_zero_count = 0

        # 将经过RLC编码的（pre_zero, value）中的value进一步用VLI变长方式编码。
        vli_rlc_list = list()
        for item in rlc_list:
            value = item[1]
            vli_tuple = self.VLI_encode(value)
            vli_rlc_list.append((item[0], vli_tuple[0], vli_tuple[1]))
        return rlc_list, vli_rlc_list

    def to_string(self, type, h, w):

        # 有VLI编码的二进制形式
        if type == 'bit-vli':
            result = self.get_intermediate(type='vli')
            index = self.get_index(h, w)
            block_result = result[index]
            bits_str = ''
            # DC
            bits_str += '{:0>4b}.'.format(block_result[0][0]) + block_result[0][1]
            # AC
            for i in range(1, len(block_result)):
                bits_str += ' {:0>4b}.{:0>4b}.'.format(block_result[i][0], block_result[i][1]) + block_result[i][2]
            return bits_str

        # 无VLI编码的二进制形式
        elif type == 'bit-raw':
            result = self.get_intermediate(type='raw')
            index = self.get_index(h, w)
            block_result = result[index]
            bits_str = ''
            # DC
            if block_result[0] >= 0:
                bits_str += '{:0>8b}'.format(block_result[0])
            else:
                bits_str += '1{:0>7b}'.format(abs(block_result[0]))
            # AC
            for i in range(1, len(block_result)):
                if block_result[i][1] >= 0:
                    bits_str += ' {:0>4b}.{:0>8b}'.format(block_result[i][0], block_result[i][1])
                else:
                    bits_str += ' {:0>4b}.1{:0>7b}'.format(block_result[i][0], abs(block_result[i][1]))
            return bits_str

        # 无VLI变长编码的中间码
        elif type == 'middle-raw':
            result = self.get_intermediate(type='raw')
            return result[self.get_index(h, w)]

        # 有VLI变长编码的中间码
        elif type == 'middle-vli':
            result = self.get_intermediate(type='vli')
            return result[self.get_index(h, w)]
        else:
            raise Exception('No such format {}.'.format(type))

    def get_intermediate(self, type):
        """
        Get list of length (height * width) for tuples shaped like (4bits, 4bits, vary) / (4bits, 4bits)
        tuple formats:
            first tuple is for DC: (row in VLI table, index)
            other tuple is for AC:
                (pre-zero-num, row in VLI table, index)
                (0, 0) End of an array.
        """
        result = list()
        if type == 'vli':
            dc = self.vli_rel_dc
            ac = self.vli_ac_rlc
        elif type == 'raw':
            dc = self.rel_dc
            ac = self.ac_rlc
        else:
            raise Exception('No such intermediate code {}'.format(type))

        # 针对每一个Block，生成(DC, AC1, AC2, ....)的列表
        for i in range(self.dct_compress_matrix.shape[0]):
            block_result = list()
            block_result.append(dc[i])
            for unit in ac[i]:
                block_result.append(unit)
            result.append(block_result)
        return result

    def get_index(self, h, w):
        return h * self.grid_w + w

    @staticmethod
    def VLI_encode(signal):
        x = 100
        if signal == 0:
            return 0, ''
        else:
            row = int(np.log2(abs(signal))) + 1
            if signal > 0:
                index = str(bin(signal))
            else:
                t = str(bin(abs(signal)))
                t = t.replace('1', 'x')
                t = t.replace('0', '1')
                index = t.replace('x', '0')
        return row, index[2:]

    @staticmethod
    def VLI_decode(row, index):
        if row == 0:
            return 0
        if index[0] == '1':
            return int('0b' + index, base=2)
        else:
            t = index
            t = t.replace('1', 'x')
            t = t.replace('0', '1')
            index = t.replace('x', '0')
            return -int('0b' + index, base=2)

    @staticmethod
    def zig_zag(block, block_h, block_w):
        # zig-zag 编码
        result = list()
        count = block_w * block_h - 1
        x, y = 0, 1  # start with fisrt AC (except DC)
        direction = 'down'  # up or down
        for cnt in range(count):
            result.append(block[x, y])
            if direction == 'up':
                x = x - 1
                y = y + 1
            else:
                x = x + 1
                y = y - 1

            # 越界，进行方向切换和下标挪移y
            if not (0 <= x < block_h and 0 <= y < block_w):
                if direction == 'up':
                    x, y = (x + 1, y) if y < block_w else (x + 2, y - 1)
                else:
                    x, y = (x, y + 1) if x < block_h else (x - 1, y + 2)
                direction = 'up' if direction == 'down' else 'down'
        return result


if __name__ == '__main__':
    print(DpcmRlcEncoder.VLI_encode(0))
    print(DpcmRlcEncoder.VLI_encode(-6))
    print(DpcmRlcEncoder.VLI_encode(17))
    print(DpcmRlcEncoder.VLI_encode(-8))
    print(DpcmRlcEncoder.VLI_decode(0, ''))
    print(DpcmRlcEncoder.VLI_decode(5, '00000'))
    print(DpcmRlcEncoder.VLI_decode(4, '0001'))
    print(DpcmRlcEncoder.VLI_decode(4, '1001'))
    demp_matrix = np.array([
        [9, 0, -1, 0, 0, 0, 0, 0],
        [-7, -1, 0, 0, 0, 0, 0, 0],
        [-1, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ])
    print(DpcmRlcEncoder.zig_zag(demp_matrix, 8, 8))
    demo = np.zeros((2, 2, 8, 8), dtype=int)
    demo[0, 0] = demp_matrix
    demo[0, 1] = demp_matrix
    demo[1, 0] = demp_matrix
    demo[1, 1] = demp_matrix

    encoder = DpcmRlcEncoder(demo)
    print(encoder.to_string('middle-raw', 0, 0))
    print(encoder.to_string('bit-raw', 0, 0))
    print(encoder.to_string('middle-vli', 0, 0))
    print(encoder.to_string('bit-vli', 0, 0))
