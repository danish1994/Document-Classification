import numpy as np

from sklearn.naive_bayes import GaussianNB

X = np.array([[-3,7],[1,5], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])
Y = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])

model = GaussianNB()

model.fit(X, Y)

predicted= model.predict([[1,2],[3,4]])
print (predicted)