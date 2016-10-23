import nltk
import nltk.data
import itertools
import nltk.tag as tagger
from nltk.chunk.named_entity import NEChunkParserTagger, NEChunkParser
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import conll2000
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.probability import FreqDist


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


def initialize(path):
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


def getNodes(parent):
    label_count = 0
    total_count = 0
    for node in parent:
        if type(node) is nltk.Tree:
            if(node.label() in ['PERSON','ORGANIZATION']):
                label_count += 1
            total_count += 1
            getNodes(node)

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

    # word_tag_fd = FreqDist(after_tag[0])
    # print([wt[0] for (wt, _) in word_tag_fd.most_common() if wt[1] == 'NN'])

    return int((total_count / label_count) * 100)
    
# train_sents = conll2000.chunked_sents('train.txt')
# test_sents = conll2000.chunked_sents('test.txt')
# print(train_sents)

# chunker = ConsecutiveNPChunker(train_sents)
# print(chunker.evaluate(test_sents))

# sentence, stemmed = zip(*initialize("vocab.txt"))
# sentence, stemmed = zip(*initialize("DataSet/Fiction/Romantic/Pride-and-Prejudice.txt"))
sentence, stemmed = zip(*initialize("DataSet/Fiction/Drama/As-Skies-Became-Crimson.txt"))
# sentence, stemmed = zip(*initialize("DataSet/Fiction/Drama/Huey.txt"))
# sentence, stemmed = zip(*initialize("DataSet/Fiction/Drama/Tanya.txt"))
# sentence, stemmed = zip(*initialize("DataSet/Fiction/Drama/The-Day-God-Came-to-Earth.txt"))
# sentence,stemmed = zip(*initialize("DataSet/Fiction/Drama/The-Diaries-of-Bunty-Danvers.txt"))
sentence = ''.join(sentence)
stemmed = ''.join(stemmed)

# Criteria 1
criteria_1 = first_criteria(stemmed)
print(criteria_1)

# Criteria 2
criteria_2 = second_criteria(stemmed)
print(criteria_2)

# Criteria 3
criteria_3 = third_criteria(sentence)
print(criteria_3)
