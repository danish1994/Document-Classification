import os
import nltk
import itertools
import nltk.tag as tagger
import numpy as np

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from utility import pdf_to_text
from tqdm import tqdm


# Shame of X Matrix
x_shape = 6


# Get X Matrix
def get_X(path):

    sentence, stemmed = zip(*get_sentence(path, True))
    sentence = ''.join(sentence)
    stemmed = ''.join(stemmed)

    print('Calculating Classification Criteria\'s')

    pbar = tqdm(total=100)

    # Defining Counts
    total_count = 0
    fdist = FreqDist()
    for x in nltk.tokenize.sent_tokenize(stemmed):
        for word in nltk.tokenize.word_tokenize(x):
            fdist[word] += 1
            total_count += 1

    pbar.update(40)

    # Criteria 1
    criteria_1 = first_criteria(total_count, fdist)

    pbar.update(5)

    # Criteria 2
    criteria_2 = second_criteria(total_count, fdist)

    pbar.update(5)

    # Criteria 3
    criteria_3 = third_criteria(sentence) / 10

    pbar.update(30)

    # Criteria 4
    criteria_4 = fourth_criteria(total_count)

    pbar.update(5)

    # Criteria 5
    criteria_5 = fifth_criteria(total_count, fdist)

    pbar.update(5)

    # Criteria 6
    criteria_6 = sixth_criteria(total_count, fdist)

    pbar.update(5)

    # Criteria 7
    criteria_7 = seventh_criteria(total_count, fdist)

    pbar.update(5)

    x = np.zeros(shape=(1, x_shape), dtype=int)
    x[0] = [criteria_1, criteria_2, criteria_3, criteria_4, criteria_5, criteria_6]

    pbar.close()

    return x


# Get Y Matrix
def get_Y(path, genres):
    file_name = path.split("/")

    genre = file_name[-2]
    genre_number = genres.index(genre)

    y = np.zeros(shape=(1, len(genres)), dtype=int)

    y[0][genre_number] = 1

    return y


# Return Result of Criteria's.
def get_criteria(path, genres):

    x = get_X(path)
    y = get_Y(path, genres)

    print(x[0])
    print(y[0])

    return zip(x, y)


# Get Senetence from File removing Stop Words and Stemming.
def get_sentence(path, show):

    file_name = path.split("/")
    file_format = file_name[-1].split('.')[-1]
    book_name = ' '.join(file_name[-1].split('.')[0].split('-')).title()

    sentence = ''

    if(file_format == 'pdf'):
        sentence = pdf_to_text(path)
    else:
        if(show):
            print('\n\nReading "' + book_name + '"')
            pbar = tqdm(total=2)
            f = open(path)
            pbar.update(1)
            sentence = f.read()
            pbar.update(1)
            pbar.close()
        else:
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
def get_nodes(parent):
    label_count = 0
    total_count = 0
    for node in parent:
        if type(node) is nltk.Tree:
            if(node.label() in ['PERSON']):
                label_count += 1
            total_count += 1
            x = get_nodes(node)
            label_count += x[0]
            total_count += x[1]

    res = [label_count, total_count]

    return res


# Get Word Count
def get_word_count(fdist, path):
    count = 0

    sentence, stemmed = zip(*get_sentence(path, False))
    stemmed = ''.join(stemmed)
    stemmed = stemmed.split(' ')

    for x in stemmed:
        count += fdist[x]

    return count


# Ratio of "I", "WE" and "You" in the Document.
def first_criteria(total_count, fdist):
    count = fdist['i'] + fdist['I'] + fdist['we'] + \
        fdist['We'] + fdist['you'] + fdist['You']

    return int((count / total_count) * 100)


# Ratio of Punctuation Marks in the Document.
def second_criteria(total_count, fdist):
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
        y = get_nodes(x)
        label_count += y[0]
        total_count += y[1]

    return int((label_count / total_count) * 100)


# Total Word Count
def fourth_criteria(total_count):
    return total_count / 1000


# Count Words Present in Thriller
def fifth_criteria(total_count, fdist):

    count = get_word_count(fdist, 'Criteria/Thriller.txt')
    return int((count / total_count) * 1000)


# Count Words Present in Romantic
def sixth_criteria(total_count, fdist):

    count = get_word_count(fdist, 'Criteria/Romantic.txt')
    return int((count / total_count) * 1000)


# Count Words Present in Drama
def seventh_criteria(total_count, fdist):

    count = get_word_count(fdist, 'Criteria/Drama.txt')
    return int((count / total_count) * 10000)
