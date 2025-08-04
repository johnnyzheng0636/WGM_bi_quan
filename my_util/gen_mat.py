# gen the saddle like distribution matrix for test

import numpy as np
import torch
from .visualization import plot_mat_dist_bar

class gen_saddle_matrix():
    def __init__(self, 
                 shape,
                 right_mean=1.0,
                 left_mean=-1.0,
                 std=1.0,
                 seed=42,
                 plotFlag=False,
                 ):
        '''
        shape: (row, col) shape of the matrix to be generated
        '''
        self.plotFlag = plotFlag

        self.seed = seed

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.right_mean = right_mean
        self.left_mean = left_mean
        self.std = std
        self.shape = shape
        self.total_size = shape[0] * shape[1]
        self.right_size = self.total_size // 2
        self.left_size = self.total_size - self.right_size
        # print('right size: ', self.right_size)
        # print('left size: ', self.left_size)
        # print('total size: ', self.total_size)
        # print('shape: ', self.shape)
        # print('right mean: ', self.right_mean)
        # print('left mean: ', self.left_mean)

        self.right_eles_freq_path = './fig/right_eles_freq.png'
        self.left_eles_freq_path = './fig/left_eles_freq.png'
        self.all_eles_freq_path = './fig/all_eles_freq.png'

        self.matrix = self.generate()

    def generate(self):
        # generate two normal distribution list with half
        # of total number of elements, make two of them
        # sysmetric about 0. Concatenate them and reshape
        # for the final matrix
        right_eles = torch.normal(self.right_mean, self.std, size=(self.right_size,))
        # plot_mat_dist_bar(right_eles, self.right_eles_freq_path)
        left_eles = torch.normal(self.left_mean, self.std, size=(self.left_size,))
        # plot_mat_dist_bar(left_eles, self.left_eles_freq_path)
        all_eles = torch.cat((right_eles, left_eles), dim=0)
        if self.plotFlag:
            plot_mat_dist_bar(all_eles, self.all_eles_freq_path)
        all_eles = all_eles.reshape(self.shape)
        return all_eles
        
