import os
import math
import numpy as np
import matplotlib.pyplot as plt

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from numpy import ones, vstack
from numpy.linalg import lstsq
from criteria import get_X as criteria_get_X
from criteria import x_shape


# Get Equation of Line.
def get_line_equation(x1, y1, x2, y2):
    points = [(x1, y1), (x2, y2)]
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords, ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords)[0]
    # print('Line Solution is y = {m}x + {c}'.format(m=round(m, 2), c=round(c, 2)))


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
def plot_subfigure(X, Y, title, transform, genres):
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

    plt.scatter(X[:, 0], X[:, 1], s=80, c='gray',
                label='Test Data')

    for i in range(0, width):
        try:
            plt.scatter(X[np.where(Y[:, i]), 0], X[np.where(
                Y[:, i]), 1], s=80, c=plot_color(i), label=genres[i])
            plot_hyperplane(classif.estimators_[i], min_x, max_x, plot_marker(
                i), 'Boundary\nfor ' + genres[i])
        except:
            plt.scatter(X[np.where(Y[:, i]), 0], X[np.where(
                Y[:, i]), 1], s=80, c=plot_color(i), label='Class ' + str(i))
            plot_hyperplane(classif.estimators_[i], min_x, max_x, plot_marker(
                i), 'Boundary\nfor Class ' + str(i))

    plt.xticks(())
    plt.yticks(())

    plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
    plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# Classify Genre
def classify_genre(X, Y, transform, genres):
    if transform == "pca":
        X = PCA(n_components=2).fit_transform(X)
    elif transform == "cca":
        X = CCA(n_components=2).fit(X, Y).transform(X)
    else:
        raise ValueErrorS

    width = Y.shape[1]

    genre = get_genre(X, Y, genres)

    return genre


# Intialize Classification
def classify(X, Y, show):

    transform = 'cca'

    genres = []
    rootdir = os.getcwd() + '/DataSet'
    for subdir, dirs, files in os.walk(rootdir):
        if len(dirs) > 1:
            genres = dirs
            break

    if(show):
        plt.figure(figsize=(10, 6))

        plot_subfigure(X, Y, "Plot Graph", transform, genres)

        plt.subplots_adjust(.07, .07, .70, .90, .09, .2)

        plt.show()

    else:
        genre = classify_genre(X, Y, transform, genres)

        return genre


# Get Matrix X
def get_X(content):
    x_shape = int(content[0])

    matrix_x = np.zeros(shape=(0, x_shape), dtype=int)

    for a in content[2].split('|'):
        temp = np.ndarray(shape=(1, x_shape), dtype=int)
        temp[0] = [int(s) for s in a.split(',')]
        matrix_x = np.concatenate((matrix_x, temp), axis=0)

    return matrix_x


# Get Matrix Y
def get_Y(content):
    y_shape = int(content[1])

    matrix_y = np.zeros(shape=(0, y_shape), dtype=int)

    for a in content[3].split('|'):
        temp = np.ndarray(shape=(1, y_shape), dtype=int)
        temp[0] = [int(s) for s in a.split(',')]
        matrix_y = np.concatenate((matrix_y, temp), axis=0)

    return matrix_y


# Show Trained Data (as a Plot).
def show_trained_data():
    content = read_trained_data()

    matrix_x = get_X(content)
    matrix_y = get_Y(content)

    classify(matrix_x, matrix_y, True)


# Read Training Data from File.
def read_trained_data():
    f = open('trained_set.txt')
    content = f.read().split('\n')

    return content


# Save Training Result to File.
def save_trained_data(matrix_x, matrix_y):
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


# Add Test Data to Trained Data.
def add_test_data(x, show):
    content = read_trained_data()

    matrix_x = get_X(content)
    matrix_y = get_Y(content)

    y = np.zeros(shape=(x.shape[0], int(content[1])), dtype=int)

    matrix_x = np.concatenate((matrix_x, x), axis=0)
    matrix_y = np.concatenate((matrix_y, y), axis=0)

    genre = classify(matrix_x, matrix_y, show)

    return genre


# Define Genre of Test Data
def get_genre(X, Y, genres):
    x_test = X[:, 0][-1]
    y_test = X[:, 1][-1]

    distances = []

    for i in range(0, len(genres)):
        x_genre = X[np.where(Y[:, i]), 0][0]
        y_genre = X[np.where(Y[:, i]), 1][0]

        sum = 0
        for j in range(0, len(x_genre)):
            sum += math.hypot(x_genre[j] - x_test, y_genre[j] - y_test)

        distances.append(sum / len(x_genre))

    i = distances.index(min(distances))

    return genres[i]


# Get Result and Accuracy from Test Data
def test_data():

    total = 0
    correct = 0

    matrix_x = np.zeros(shape=(0, x_shape), dtype=int)

    rootdir = os.getcwd() + '/TestData'

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            path = os.path.join(subdir, file)

            file_name = path.split("/")
            genre = file_name[-2]

            book_name = ' '.join(
                file_name[-1].split('.')[0].split('-')).title()

            x = criteria_get_X(path)
            matrix_x = np.concatenate((matrix_x, x), axis=0)

            calculated_genre = add_test_data(x, False)

            print(book_name + ' - ' + calculated_genre)

            if(calculated_genre == genre):
                print('Right Answer')
                correct += 1
            else:
                print('Wrong Answer')
            total += 1

    accuracy = round((correct / total) * 100, 2)

    print('\nRight = \t{correct}'.format(correct=correct))
    print('Total = \t{total}'.format(total=total))

    print('Accuracy = \t{accuracy}%.\n'.format(
        accuracy=accuracy))

    add_test_data(matrix_x, True)


# Test Individual File
def test_single_data(path):
    matrix_x = np.zeros(shape=(0, x_shape), dtype=int)

    file_name = path.split("/")
    book_name = ' '.join(file_name[-1].split('.')[0].split('-'))

    x = criteria_get_X(path)
    matrix_x = np.concatenate((matrix_x, x), axis=0)

    calculated_genre = add_test_data(x, False)

    print(book_name + ' - ' + calculated_genre)

    add_test_data(x, True)
