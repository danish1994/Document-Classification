import os
import numpy as np

from criteria import get_criteria
from classify import show_trained_data
from classify import save_trained_data
from criteria import x_shape



# Generate List of Genres
genres = []
rootdir = os.getcwd() + '/DataSet'
for subdir, dirs, files in os.walk(rootdir):
    if len(dirs) > 1:
        genres = dirs
        break


# Intitalizing Result Matrix for MatPlot.
matrix_x = np.zeros(shape=(0, x_shape), dtype=int)
matrix_y = np.zeros(shape=(0, len(genres)), dtype=int)


# Iterating Through DataSets
rootdir = os.getcwd() + '/DataSet'
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        path = os.path.join(subdir, file)

        x, y = zip(*get_criteria(path, genres))

        matrix_x = np.concatenate((matrix_x, x), axis=0)
        matrix_y = np.concatenate((matrix_y, y), axis=0)


save_trained_data(matrix_x, matrix_y)
show_trained_data()
