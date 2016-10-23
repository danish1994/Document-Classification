import os
import numpy as np
from retrieval import get_criteria


# Save Training Result to File.
def save_to_file(matrix_x, matrix_y):
	matrix_x_str = ''
	for a in matrix_x:
		matrix_x_str += ','.join(str(e) for e in a)
		matrix_x_str += '|'

	matrix_y_str = ''
	for a in matrix_y:
		matrix_y_str += ','.join(str(e) for e in a)
		matrix_y_str += '|'

	matrix_x_str = matrix_x_str[:-1]
	matrix_y_str = matrix_y_str[:-1]

	final_str = str(matrix_x.shape[1]) + '\n' + str(matrix_y.shape[1]) + '\n' + matrix_x_str + '\n' + matrix_y_str

	f = open('trained_set.txt', 'w')
	f.write(final_str)


# Intitalizing Result Matrix for MatPlot.
matrix_x = np.zeros(shape=(0, 3), dtype=int)
matrix_y = np.zeros(shape=(0, 3), dtype=int)

# Iterating Through DataSets
rootdir = os.getcwd() + '/DataSet'
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        path = os.path.join(subdir, file)

        x, y = zip(*get_criteria(path))

        matrix_x = np.concatenate((matrix_x, x), axis=0)
        matrix_y = np.concatenate((matrix_y, y), axis=0)

save_to_file(matrix_x, matrix_y)
# classify(matrix_x, matrix_y)
