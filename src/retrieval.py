# from Natural Language Processing with Python book.
from nltk.corpus import conll2000
import nltk
import nltk.tag as tagger

class UnigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)] for sent in train_sents]
        self.tagger = nltk.UnigramTagger(train_data)

    def parse(self, sentence):
        pos_tags = [pos for (word, pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word, pos), chunktag) in zip(sentence, chunktags)]
        return nltk.chunk.util.conlltags2tree(conlltags)

def npchunk_features(sentence, i, history):
    word, pos = sentence[i]
    if i == 0:
       prevword, prevpos = "<START>", "<START>"
    else:
       prevword, prevpos = sentence[i-1]
    return {"pos": pos, "prevpos": prevpos}

class ConsecutiveNPChunkTagger(nltk.TaggerI):

    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history)
                train_set.append( (featureset, tag) )
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
        tagged_sents = [[((w,t),c) for (w,t,c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)

train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])

sentence = "A mobile phone is a portable telephone that can make and receive calls over a radio frequency carrier while the user is moving within a telephone service area. The radio frequency link establishes a connection to the switching systems of a mobile phone operator, which provides access to the public switched telephone network (PSTN). Most modern mobile telephone services use a cellular network architecture, and therefore mobile telephones are often also called cellular telephones or cell phones. In addition to telephony, 2000s-era mobile phones support a variety of other services, such as text messaging, MMS, email, Internet access, short-range wireless communications (infrared, Bluetooth), business applications, gaming, and digital photography. Mobile phones which offer these and more general computing capabilities are referred to as smartphones."
seent = nltk.sent_tokenize(sentence)
tagged = [nltk.word_tokenize(se) for se in seent]
after_tag = [nltk.pos_tag(ta) for ta in tagged]

chunker = ConsecutiveNPChunker(train_sents)
print(chunker.evaluate(test_sents))

for tag in after_tag:
    zipfile = chunker.tagger.tag(tag)
    sentence, history= zip(*zipfile)

    chunker.parse(sentence).draw()