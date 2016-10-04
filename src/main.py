import numpy as np

from classify import classify

X = np.array([
        [1, 1],
        [2, 1],
        [2, 5],
        [2, 2]
        ], np.int32)

Y = np.array([
	    [0, 1],
        [1, 1],
        [1, 0],
        [0, 0]
        ], np.int32)

classify(X,Y)