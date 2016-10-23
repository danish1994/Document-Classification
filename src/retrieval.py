import nltk
import nltk.data
import itertools
import os
import nltk.tag as tagger
import numpy as np
from nltk.chunk.named_entity import NEChunkParserTagger, NEChunkParser
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import conll2000
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.probability import FreqDist
from classify import classify


class UnigramChunker(nltk.ChunkParserI):

    def __init__(self, train_sents):
        train_data = [[(t, c) for w, t, c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.UnigramTagger(train_data)

    def parse(self, sentence):
        pos_tags = [pos for (word, pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag)
                     for ((word, pos), chunktag) in zip(sentence, chunktags)]
        return nltk.chunk.util.conlltags2tree(conlltags)


class ConsecutiveNPChunkTagger(nltk.TaggerI):

    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history)
                train_set.append((featureset, tag))
                history.append(tag)
        self.classifier = nltk.MaxentClassifier.train(
            train_set, algorithm='IIS', trace=0)
        self.classifier.show_most_informative_features(20)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)


class ConsecutiveNPChunker(nltk.ChunkParserI):

    def __init__(self, train_sents):
        tagged_sents = [[((w, t), c) for (w, t, c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w, t, c) for ((w, t), c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)


class TagChunker(nltk.chunk.ChunkParserI):

    def __init__(self, chunk_tagger):
        self._chunk_tagger = chunk_tagger

    def parse(self, tokens):
        # split words and part of speech tags
        (words, tags) = zip(*tokens)
        # get IOB chunk tags
        chunks = self._chunk_tagger.tag(tags)
        # join words with chunk tags
        wtc = itertools.izip(words, chunks)
        # w = word, t = part-of-speech tag, c = chunk tag
        lines = [' '.join([w, t, c]) for (w, (t, c)) in wtc if c]
        # create tree from conll formatted chunk lines
        return nltk.chunk.conllstr2tree('\n'.join(lines))


def npchunk_features(sentence, i, history):
    word, pos = sentence[i]
    if i == 0:
        prevword, prevpos = "<START>", "<START>"
    else:
        prevword, prevpos = sentence[i - 1]
    return {"pos": pos, "prevpos": prevpos}


# def npchunk_features(sentence, i, history):
#     word, pos = sentence[i]
#     if i == 0:
#         prevword, prevpos = "<START>", "<START>"
#     else:
#         prevword, prevpos = sentence[i - 1]
#     if i == len(sentence) - 1:
#         nextword, nextpos = "<END>", "<END>"
#     else:
#         nextword, nextpos = sentence[i + 1]
#     return {"pos": pos,
#             "word": word,
#             "prevpos": prevpos,
#             "nextpos": nextpos,
#             "prevpos+pos": "%s+%s" % (prevpos, pos),
#             "pos+nextpos": "%s+%s" % (pos, nextpos),
#             "tags-since-dt": tags_since_dt(sentence, i)}


# def tags_since_dt(sentence, i):
#     tags = set()
#     for word, pos in sentence[:i]:
#         if pos == 'DT':
#             tags = set()
#         else:
#             tags.add(pos)
#     return '+'.join(sorted(tags))

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
    criteria_3 = third_criteria(sentence)/10

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


# train_sents = conll2000.chunked_sents('train.txt')
# test_sents = conll2000.chunked_sents('test.txt')
# # stemmer = SnowballStemmer("english")
# f = open("vocab.txt")
# f1 = open("finalVocab.txt","w")
# sentence = f.read()
# stop = set(stopwords.words('english'))
# sentence = [i for i in sentence.split() if i not in stop]
# # plurals = [stemmer.stem(sente) for sente in sentence]
# print(sentence)
# # print(plurals)
# sentence = ' '.join(str(e) for e in sentence)
# # plurals = ' '.join(str(e) for e in plurals)
# # print(plurals)
# seent = nltk.sent_tokenize(sentence)
# tagged = [nltk.word_tokenize(se) for se in seent]
# after_tag = [nltk.pos_tag(ta) for ta in tagged]

# chunker = ConsecutiveNPChunker(train_sents)
# print(chunker.evaluate(test_sents))

# senti = after_tag[0]
# print(nltk.ne_chunk(senti).draw())

# for tag in after_tag:
#     tree = chunker.tagger.tag(tag)
#     sentence, history= zip(*tree)
#     for i,j in zip(sentence,history):
#         print(i[0],i[1],j)
#         s = i[0]+' '+i[1]+' '+j
#         f1.write(s+"\n")
#     chunker.parse(sentence)