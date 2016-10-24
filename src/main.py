import numpy as np
import nltk

from classify import read_from_file
from classify import classify
from classify import test_data

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

test_x = np.zeros(shape=(1, 6), dtype=int)
test_x[0] = [0, 20, 1, 6, 0,19]

# test_data(test_x)

read_from_file()
# classify(X, Y)
