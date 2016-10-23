import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from numpy import ones, vstack
from numpy.linalg import lstsq


# Get Equation of Line.
def get_line_equation(x1, y1, x2, y2):
    points = [(x1, y1), (x2, y2)]
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords, ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords)[0]
    print(
        "Line Solution is y = {m}x + {c}".format(m=round(m, 2), c=round(c, 2)))


# PLot the HyperPlane Classifying the Data.
def plot_hyperplane(clf, min_x, max_x, linestyle, label):
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]
    get_line_equation(xx[0], yy[0], xx[1], yy[1])
    plt.plot(xx, yy, linestyle, label=label)


# Colour of the Point.
def plot_color(i):
    colors = {
        0: 'b',
        1: 'g',
        2: 'r',
        3: 'orange'
    }
    return colors.get(i, 'b')


# Style of the HyperPlane.
def plot_marker(i):
    colors = {
        0: 'k-',
        1: 'k--',
        2: 'k-.',
        3: ':'
    }
    return colors.get(i, 'k-')


# Plot Subfigure.
def plot_subfigure(X, Y, title, transform):
    if transform == "pca":
        X = PCA(n_components=2).fit_transform(X)
    elif transform == "cca":
        X = CCA(n_components=2).fit(X, Y).transform(X)
    else:
        raise ValueError

    min_x = np.min(X[:, 0])
    max_x = np.max(X[:, 0])

    min_y = np.min(X[:, 1])
    max_y = np.max(X[:, 1])

    classif = OneVsRestClassifier(SVC(kernel='linear'))
    classif.fit(X, Y)

    plt.title(title)

    width = Y.shape[1]

    for i in range(0, width):
        plt.scatter(X[np.where(Y[:, i] == 1), 0], X[np.where(
            Y[:, i] == 1), 1], s=80, c=plot_color(i), label='Class ' + str(i))
        plot_hyperplane(classif.estimators_[i], min_x, max_x, plot_marker(
            i), 'Boundary\nfor class ' + str(i))

    plt.xticks(())
    plt.yticks(())

    plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
    plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# Intialize Classification
def classify(X, Y):
    plt.figure(figsize=(8, 6))

    plot_subfigure(X, Y, "Plot Graph", "pca")

    plt.subplots_adjust(.07, .07, .70, .90, .09, .2)

    plt.show()


# Read Training Data from File.
def read_from_file(path):
    f = open(path)
    content = f.read().split('\n')

    x_shape = int(content[0])
    y_shape = int(content[1])

    matrix_x = np.zeros(shape=(0, x_shape), dtype=int)
    matrix_y = np.zeros(shape=(0, y_shape), dtype=int)

    for a in content[2].split('|'):
        temp = np.ndarray(shape=(1, x_shape), dtype=int)
        temp[0] = [int(s) for s in a.split(',')]
        matrix_x = np.concatenate((matrix_x, temp), axis=0)

    for a in content[3].split('|'):
        temp = np.ndarray(shape=(1, y_shape), dtype=int)
        temp[0] = [int(s) for s in a.split(',')]
        matrix_y = np.concatenate((matrix_y, temp), axis=0)

    classify(matrix_x, matrix_y)


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

    final_str = str(matrix_x.shape[
                    1]) + '\n' + str(matrix_y.shape[1]) + '\n' + matrix_x_str + '\n' + matrix_y_str

    f = open('trained_set.txt', 'w')
    f.write(final_str)
