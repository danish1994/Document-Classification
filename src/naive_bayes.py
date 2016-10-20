
###################
# Version 1 Start #
###################

# import numpy as np

# from sklearn.naive_bayes import GaussianNB

# X = np.array([[-3,7],[1,5], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])
# Y = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])

# model = GaussianNB()

# model.fit(X, Y)

# predicted= model.predict([[1,2],[3,4]])
# print (predicted)

#################
# Version 1 End #
#################


###################
# Version 2 Start #
###################

from sklearn import datasets
iris = datasets.load_iris()
print(iris)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d" %
      (iris.data.shape[0], (iris.target != y_pred).sum()))

#################
# Version 2 End #
#################


###################
# Version 3 Start #
###################

# from collections import defaultdict

# from nltk.probability import FreqDist, DictionaryProbDist, ELEProbDist, sum_logs
# from nltk.classify.api import ClassifierI


# class NaiveBayesClassifier(ClassifierI):
#     def __init__(self, label_probdist, feature_probdist):
#         self._label_probdist = label_probdist
#         self._feature_probdist = feature_probdist
#         self._labels = list(label_probdist.samples())

#     def labels(self):
#         return self._labels

#     def classify(self, featureset):
#         return self.prob_classify(featureset).max()

#     def prob_classify(self, featureset):
#         featureset = featureset.copy()
#         for fname in list(featureset.keys()):
#             for label in self._labels:
#                 if (label, fname) in self._feature_probdist:
#                     break
#             else:
#                 del featureset[fname]

#         logprob = {}
#         for label in self._labels:
#             logprob[label] = self._label_probdist.logprob(label)

#         for label in self._labels:
#             for (fname, fval) in featureset.items():
#                 if (label, fname) in self._feature_probdist:
#                     feature_probs = self._feature_probdist[label, fname]
#                     logprob[label] += feature_probs.logprob(fval)
#                 else:
#                     logprob[label] += sum_logs([]) # = -INF.

#         return DictionaryProbDist(logprob, normalize=True, log=True)

#     def show_most_informative_features(self, n=10):
#         cpdist = self._feature_probdist
#         print('Most Informative Features')

#         for (fname, fval) in self.most_informative_features(n):
#             def labelprob(l):
#                 return cpdist[l, fname].prob(fval)

#             labels = sorted([l for l in self._labels
#                              if fval in cpdist[l, fname].samples()],
#                             key=labelprob)
#             if len(labels) == 1:
#                 continue
#             l0 = labels[0]
#             l1 = labels[-1]
#             if cpdist[l0, fname].prob(fval) == 0:
#                 ratio = 'INF'
#             else:
#                 ratio = '%8.1f' % (cpdist[l1, fname].prob(fval) /
#                                    cpdist[l0, fname].prob(fval))
#             print(('%24s = %-14r %6s : %-6s = %s : 1.0' %
#                    (fname, fval, ("%s" % l1)[:6], ("%s" % l0)[:6], ratio)))

#     def most_informative_features(self, n=100):
#         features = set()
#         maxprob = defaultdict(lambda: 0.0)
#         minprob = defaultdict(lambda: 1.0)

#         for (label, fname), probdist in self._feature_probdist.items():
#             for fval in probdist.samples():
#                 feature = (fname, fval)
#                 features.add(feature)
#                 p = probdist.prob(fval)
#                 maxprob[feature] = max(p, maxprob[feature])
#                 minprob[feature] = min(p, minprob[feature])
#                 if minprob[feature] == 0:
#                     features.discard(feature)

#         features = sorted(features,
#                           key=lambda feature_:
#                           minprob[feature_]/maxprob[feature_])
#         return features[:n]

#     @classmethod
#     def train(cls, labeled_featuresets, estimator=ELEProbDist):
#         label_freqdist = FreqDist()
#         feature_freqdist = defaultdict(FreqDist)
#         feature_values = defaultdict(set)
#         fnames = set()

#         for featureset, label in labeled_featuresets:
#             label_freqdist[label] += 1
#             for fname, fval in featureset.items():
#                 feature_freqdist[label, fname][fval] += 1
#                 feature_values[fname].add(fval)
#                 fnames.add(fname)

#         for label in label_freqdist:
#             num_samples = label_freqdist[label]
#             for fname in fnames:
#                 count = feature_freqdist[label, fname].N()
#                 if num_samples - count > 0:
#                     feature_freqdist[label, fname][None] += num_samples - count
#                     feature_values[fname].add(None)

#         label_probdist = estimator(label_freqdist)

#         feature_probdist = {}
#         for ((label, fname), freqdist) in feature_freqdist.items():
#             probdist = estimator(freqdist, bins=len(feature_values[fname]))
#             feature_probdist[label, fname] = probdist

#         return cls(label_probdist, feature_probdist)

# def demo():
#     from nltk.classify.util import names_demo
#     classifier = names_demo(NaiveBayesClassifier.train)
#     classifier.show_most_informative_features()

# if __name__ == '__main__':
#     demo()


#################
# Version 3 End #
#################
