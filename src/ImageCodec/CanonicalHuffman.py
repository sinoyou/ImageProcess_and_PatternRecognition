class CanonicalHuffman:
    def __init__(self, signs, sign_bit_width):
        self.signs = signs
        self.sign_bit_width = sign_bit_width
        self.vanilla_huffman_table = dict()
        self.canonical_huffman_table = dict()

        # get vanilla huffman dict
        self.vanilla_huffman_table = self.build_vanilla_table()

        # get canonical huffman dict
        self.canonical_huffman_table = self.build_canonical_table()

    def build_vanilla_table(self):
        """
        Build vanilla table with two fork tree.
        :return: dict()
        """
        # frequency analysis
        frequency = dict()
        for s in self.signs:
            frequency.setdefault(s, 0)
            frequency[s] += 1

        # run Huffman 2-fork trees generation algorithms.
        # tree node format (cost, left_index, right_index, sign(only for leaf nodes))
        nodes = list()
        trees = list()
        p = 0

        # initial
        for k, v in frequency.items():
            node = (v, None, None, k)
            nodes.insert(p, node)
            trees.append(node)
            p += 1

        # combine
        while len(trees) > 1:
            trees = sorted(trees, key=lambda x: x[0])
            a = trees.pop(0)
            b = trees.pop(0)
            new_node = (a[0] + b[0], a, b, None)
            trees.append(new_node)

        # generate vanilla huffman dictionary
        vanilla_dict = dict()
        root = trees[0]

        def dfs(node, string):
            if node[1] is None and node[2] is None:
                vanilla_dict[node[3]] = string
            elif node[1] is None or node[2] is None:
                raise Exception('Middle node should have 2 sons in huffman trees.')
            else:
                dfs(node[1], string + '0')
                dfs(node[2], string + '1')

        dfs(root, '')
        return vanilla_dict

    def build_canonical_table(self):
        """
        Build canonical table based on built vanilla table.
        Canonical Huffman Coding enable saving files without binary codes.
        :return: dict()
        """
        vanilla_items = sorted(self.vanilla_huffman_table.items(), key=lambda x: len(x[1]))
        canonical_table = dict()
        count = 0
        pre_length = 0
        pre_value = 0
        for k, v in vanilla_items:
            count += 1
            # rule 1: first binary code must be zero with same lenth
            if count == 1:
                t = ''
                pre_length = len(v)
                pre_value = 0
                canonical_table[k] = '0'.zfill(len(v))
            # rule 2: binary code with same length as previous one will be +1.
            else:
                if pre_length == len(v):
                    canonical_table[k] = '{:b}'.format(pre_value + 1).zfill(len(v))
                    pre_value += 1
                # rule：binary code with larger length, besides + 1, then << (new_len - old_len)
                else:
                    canonical_table[k] = '{:b}'.format((pre_value + 1) << (len(v) - pre_length)).zfill(len(v))
                    pre_value = (pre_value + 1) << (len(v) - pre_length)
                    pre_length = len(v)
        return canonical_table

    def get_table(self, table='canonical'):
        if table == 'canonical':
            return self.canonical_huffman_table
        elif table == 'vanilla':
            return self.vanilla_huffman_table
        else:
            raise Exception('No such huffman table type = {}'.format(table))

    def get_canonical_table_size(self):
        size = 0
        # 范式Huffman编码无需保存信号Sign所对应的二进制串，仅需要按顺序保存 <Sign, 对应二进制串长度>
        # sign
        size += len(self.canonical_huffman_table.keys()) * self.sign_bit_width
        # length, 在最坏情况下，二进制串的最大长度是符号所取得的最大值（huffman树是一个串），因此二进制串最大长度为sign_bit_width
        size += len(self.canonical_huffman_table.keys()) * self.sign_bit_width
        return size

    def encode_sign(self, sign, table='canonical'):
        if sign not in self.vanilla_huffman_table.keys():
            raise Exception('No such sign = {} in table. '.format(sign))
        if table == 'canonical':
            return self.canonical_huffman_table[sign]
        elif table == 'vanilla':
            return self.vanilla_huffman_table[sign]
        else:
            raise Exception('No such table type = {}'.format(table))


if __name__ == '__main__':
    arrays = [1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5]

    huffman = CanonicalHuffman(arrays, sign_bit_width=4)

    print(huffman.get_table(table='vanilla'))
    print(huffman.get_table(table='canonical'))
