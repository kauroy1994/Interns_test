from sentence_transformers import SentenceTransformer
from math import sqrt
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

class Math(object):

    @staticmethod
    def cossim(encoding1, encoding2):

        l1,l2 = list(encoding1),list(encoding2)
        N = len(l1)
        dotp = sum([l1[i]*l2[i] for i in range(N)])
        z1 = sqrt(sum([l1[i]**2 for i in range(N)]))
        z2 = sqrt(sum([l2[i]**2 for i in range(N)]))
        return dotp/float(z1*z2)

class NLP(object):

    @staticmethod
    def embedding(token):

        model = SentenceTransformer('all-MiniLM-L6-v2')
        encodings = model.encode([token])
        return encodings[0]
        
    @staticmethod
    def compare_sentences(sentence1, sentence2):
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        encodings = model.encode([sentence1,sentence2])
        sim = Math.cossim(encodings[0],encodings[1])
        return sim

    @staticmethod
    def pos_tags(sent):
        sent = nltk.word_tokenize(sent)
        sent = nltk.pos_tag(sent)
        return sent

class Processing(object):

    @staticmethod
    def read_file(file):

        lines = []
        with open(file,'r') as fp:
            lines = fp.read().splitlines()

        return lines

    @staticmethod
    def to_knowledge_tuple(line):

        knowledge = tuple(line[1:-1].split(','))
        return ([item.strip()[1:-1] for item in knowledge])
