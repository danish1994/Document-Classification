import os
import numpy as np
from criteria import get_criteria
from classify import classify
from classify import save_to_file


# Intitalizing Result Matrix for MatPlot.
matrix_x = np.zeros(shape=(0, 3), dtype=int)
matrix_y = np.zeros(shape=(0, 4), dtype=int)


# Generate List of Genres
genres = []
rootdir = os.getcwd() + '/DataSet'
for subdir, dirs, files in os.walk(rootdir):
    if len(dirs) > 1:
    	genres = dirs
    	break

# Iterating Through DataSets
rootdir = os.getcwd() + '/DataSet'
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        path = os.path.join(subdir, file)

        x, y = zip(*get_criteria(path, genres))

        matrix_x = np.concatenate((matrix_x, x), axis=0)
        matrix_y = np.concatenate((matrix_y, y), axis=0)

save_to_file(matrix_x, matrix_y)
# classify(matrix_x, matrix_y)
