import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from numpy import ones,vstack
from numpy.linalg import lstsq

def get_line_equation(x1, y1, x2, y2):
    points = [(x1,y1),(x2,y2)]
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords,ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords)[0]
    print("Line Solution is y = {m}x + {c}".format(m = round(m, 2), c = round(c, 2)))

def plot_hyperplane(clf, min_x, max_x, linestyle, label):
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]
    get_line_equation(xx[0], yy[0], xx[1], yy[1])
    plt.plot(xx, yy, linestyle, label=label)

def plot_color(i):
    colors = {
        0: 'b',
        1: 'g',
        2: 'r',
        3: 'orange'
    }
    return colors.get(i, 'b')

def plot_marker(i):
    colors = {
        0: 'k-',
        1: 'k--',
        2: 'k-.',
        3: ':'
    }
    return colors.get(i, 'k-')


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

    for i in range (0,width):
        plt.scatter(X[np.where(Y[:,i]==1), 0], X[np.where(Y[:,i]==1), 1], s=80, c=plot_color(i), label='Class ' + str(i))
        plot_hyperplane(classif.estimators_[i], min_x, max_x, plot_marker(i), 'Boundary\nfor class ' + str(i))

    plt.xticks(())
    plt.yticks(())

    plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
    plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


def classify(X,Y):
    plt.figure(figsize=(8, 6))

    plot_subfigure(X, Y, "Plot Graph", "cca")
    
    plt.subplots_adjust(.07, .07, .70, .90, .09, .2)

    plt.show()
