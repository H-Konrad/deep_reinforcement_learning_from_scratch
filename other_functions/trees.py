import numpy as np

class sum_tree:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size                                                      # memory size
        self.original_index_start = self.buffer_size - 1                                    # start of leaf nodes
        self.data_tree = np.full(((self.buffer_size * 2) - 1), 1e-12, dtype = np.float32)   # tree

    def add_data(self, index, priority):
        data_index = index + self.original_index_start       # where to add data
        difference = priority - self.data_tree[data_index]
        self.data_tree[data_index] = priority
        while data_index != 0:                               
            data_index = (data_index - 1) // 2         # goes down the tree until it reaches the root
            self.data_tree[data_index] += difference   # adds priority difference each time

    def derive_data(self, random_value):
        index = 0
        while index < self.original_index_start:   # find the leaf node
            left = (2 * index) + 1
            right = (2 * index) + 2
            if self.data_tree[left] >= random_value:
                index = left
            else:
                random_value -= self.data_tree[left]
                index = right

        return index - self.original_index_start, self.data_tree[index]

    def largest_value(self):
        return self.data_tree[self.original_index_start:].max()