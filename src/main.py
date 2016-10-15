import numpy as np
import nltk

from classify import classify

X = np.array([
        [1, 1],
        [2, 1],
        [2, 5],
        [2, 2]
        ], np.int32)

Y = np.array([
	[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
        ], np.int32)

classify(X,Y)