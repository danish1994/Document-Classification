from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
import nltk
import itertools
import nltk.tag as tagger
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from nltk.probability import FreqDist
import criteria

n_samples = 2000
n_features = 1000
n_topics = 10
n_top_words = 20


def get_nodes(parent):
    entity_list = []
    for node in parent:
        if type(node) is nltk.Tree:
            if(node.label() in ['PERSON', 'ORGANIZATION']):
                entity_list.append(node.leaves())
            x = get_nodes(node)

    res = entity_list

    return res


def create_entity(sentence):
    # f = open(path)
    entity_list = []
    seent = nltk.sent_tokenize(sentence)
    tagged = [nltk.word_tokenize(se) for se in seent]
    after_tag = [nltk.pos_tag(ta) for ta in tagged]
    for t in after_tag:
        x = nltk.ne_chunk(t)
        y = get_nodes(x)
        print(y)
        for tag in y:
            for i in tag:
                entity_list.append(i[0])

    return entity_list


def print_top_words(model, feature_names, n_top_words, lda_list):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        topic = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(topic)
        lda_list.append(topic)
        print()

    return lda_list

def sentence_split():
	sentence, stemmed = zip(*(criteria.get_sentence("vocab.txt")))
	sentence = ''.join(sentence)
	entity_list = create_entity(sentence)
	sentence = [i for i in sentence.split() if i not in entity_list]
	sentence = ' '.join(str(e) for e in sentence)
	sentence_list = sentence.split("......")

	return sentence_list


print("Loading dataset...")

sentence_list = sentence_split()
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer()
t0 = time()
tf = tf_vectorizer.fit_transform(sentence_list)
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=500,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
t0 = time()
lda.fit(tf)
print(lda.score(tf))
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
lda_list = []
lda_list = print_top_words(lda, tf_feature_names, n_top_words,lda_list)
lda_list = ' '.join(str(e) for e in lda_list)

