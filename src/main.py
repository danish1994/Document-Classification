import numpy as np
import nltk

from classify import read_from_file
from classify import classify
from classify import test_data
from criteria import get_X

X = np.array([
    [4, 3],
    [6, 2],
    [2, 5],
    [7, 7],
    [2, 1],
    [7, 4],
    [4, 6],
    [7, 9],
    [3, 3],
    [5, 8],
    [4, 7],
    [8, 7]
], np.int32)

Y = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 0]
], np.int32)

# x = get_X('DataSet/Fiction/Romantic/A-Walk-to-Remember.txt')

# test_data(x)

read_from_file()
# classify(X, Y)
