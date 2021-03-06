'''
This module takes care of drawing the electronic densities for the problem.
'''

import numpy as np
from quantumpropagator import chunksOfList

def give_me_swapd_oxiallyl(vector):
    '''
    this function is problem bound and it returns the correct vector where any
    AO,AO matrix has the columns/rows swapped to take into account different ordering
    between Gaussian and Molcas.

    The specific swapvector has been produced comparing gaussian and molcas label
    for the AUG-CC-PVDZ basis set of a specific molecule in a specific sequence.

    This can be "easily" generalized but for the moment, let it be...
    '''

    swapvector = [1, 2, 3, 4, 5, 8, 11, 6, 9, 12, 7, 10, 13, 18, 23, 16, 21, 14,
                  19, 15, 20, 17, 22, 24, 25, 26, 27, 28, 31, 34, 29, 32, 35, 30,
                  33, 36, 41, 46, 39, 44, 37, 42, 38, 43, 40, 45, 47, 48, 49, 50,
                  51, 54, 57, 52, 55, 58, 53, 56, 59, 64, 69, 62, 67, 60, 65, 61,
                  66, 63, 68, 70, 71, 72, 73, 74, 77, 80, 75, 78, 81, 76, 79, 82,
                  87, 92, 85, 90, 83, 88, 84, 89, 86, 91, 93, 94, 95, 96, 99, 97,
                 100, 98, 101, 102, 103, 104, 105, 108, 106, 109, 107, 110, 111,
                 112, 113, 114, 117, 115, 118, 116, 119, 120, 121, 122, 123, 126,
                 124, 127, 125, 128]
    swap_indexes = np.array(swapvector) - 1
    #num = 128 # I use this fixed numer to underline the problem/bound function
    swap_vector = vector[:, swap_indexes][swap_indexes]
    return swap_vector

def transform_numpy_into_format(array):
    '''
    Format a Numpy array into the Gaussian fchk format.
    '''
    strings = ['{:15.8E}'.format(i) for i in array]
    columns = list(chunksOfList(strings, 5))
    return '\n'.join([' '.join(x) for x in columns])

