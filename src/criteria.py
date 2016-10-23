import os
import nltk
import itertools
import nltk.tag as tagger
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.probability import FreqDist


# Return Result of Criteria's.
def get_criteria(path):
    file_name = path.split("/")
    genre = file_name[-2]

    print(file_name[-1])

    x = np.zeros(shape=(1, 3), dtype=int)
    y = np.zeros(shape=(1, 3), dtype=int)
    if (genre == 'Drama'):
        y[0] = [1, 0, 0]
    elif(genre == 'Romantic'):
        y[0] = [0, 1, 0]
    elif(genre == 'Thriller'):
        y[0] = [0, 0, 1]
    else:
        y[0] = [0, 0, 0]

    sentence, stemmed = zip(
        *get_sentence(path))
    sentence = ''.join(sentence)
    stemmed = ''.join(stemmed)

    # Criteria 1
    criteria_1 = first_criteria(stemmed)

    # Criteria 2
    criteria_2 = second_criteria(stemmed)

    # Criteria 3
    criteria_3 = third_criteria(sentence) / 10

    x[0] = [criteria_1, criteria_2, criteria_3]

    print(x[0])
    print(y[0])

    return zip(x, y)


# Get Senetence from File removing Stop Words and Stemming.
def get_sentence(path):
    f = open(path)
    sentence = f.read()
    stemmer = SnowballStemmer('english')
    stop = set(stopwords.words('english'))
    sentence = [i for i in sentence.split() if i not in stop]
    stemmed = [stemmer.stem(sente) for sente in sentence]
    sentence = ' '.join(str(e) for e in sentence)
    stemmed = ' '.join(str(e) for e in stemmed)

    res = zip(sentence, stemmed)

    return res


# Parse Through Parse Tree Nodes.
def getNodes(parent):
    label_count = 0
    total_count = 0
    for node in parent:
        if type(node) is nltk.Tree:
            if(node.label() in ['PERSON', 'ORGANIZATION']):
                label_count += 1
            total_count += 1
            x = getNodes(node)
            label_count += x[0]
            total_count += x[1]

    res = [label_count, total_count]

    return res


# Ratio of "I", "WE" and "You" in the Document.
def first_criteria(sentence):
    total_count = 0

    fdist = FreqDist()
    for seeent in nltk.tokenize.sent_tokenize(sentence):
        for word in nltk.tokenize.word_tokenize(seeent):
            fdist[word] += 1
            total_count += 1

    count = fdist['i'] + fdist['I'] + fdist['we'] + \
        fdist['We'] + fdist['you'] + fdist['You']

    return int((count / total_count) * 100)


# Ratio of Punctuation Marks in the Document.
def second_criteria(sentence):
    total_count = 0

    fdist = FreqDist()
    for seeent in nltk.tokenize.sent_tokenize(sentence):
        for word in nltk.tokenize.word_tokenize(seeent):
            fdist[word] += 1
            total_count += 1

    count = fdist["'"] + fdist[':'] + fdist[','] + fdist['-'] + fdist['...'] + \
        fdist['!'] + fdist['.'] + fdist['?'] + fdist['"'] + fdist[';']

    return int((count / total_count) * 100)


# Count of "PERSON" entity in the Document.
def third_criteria(sentence):
    seent = nltk.sent_tokenize(sentence)
    text = nltk.Text(seent)
    tagged = [nltk.word_tokenize(se) for se in seent]
    after_tag = [nltk.pos_tag(ta) for ta in tagged]

    total_count = 0

    label_count = 0

    for sentences in after_tag:
        x = nltk.ne_chunk(sentences)
        y = getNodes(x)
        label_count += y[0]
        total_count += y[1]

    return int((label_count / total_count) * 100)
