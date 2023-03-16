import unicodedata
import re
import torch
import random
MAX_LENGTH = 10
device = torch.device('cpu')
in_lang = 'eng'
out_lang = 'cha'

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub("[.!?。？！]", "\1", s)
    s = re.sub(r"[0-9]+", r" ", s)
    return s


def filterPairs(pairs):
    p = []
    for pair in pairs:
        if len(pair[0]) <= MAX_LENGTH and len(pair[1]) <= MAX_LENGTH:
            p.append(pair)

    return p


SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split():
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def readLangs(lang1, lang2, reverse=False):
    lines = open('%s-%s.txt' % (lang1, lang2),
                 encoding='utf-8').read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    for i in range(0, len(pairs)):
        del pairs[i][0:1]
        del pairs[i][1:2]

    # print(len(pairs))
    pairs = filterPairs(pairs)
    # print(len(pairs))

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    return input_lang, output_lang, pairs


def indexesFromSentence(lang, sentence):

    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


input_lang, output_lang, _ = readLangs('eng', 'cha', reverse=False)


def tensorFromPair(pair):

    input_tensor = tensorFromSentence(input_lang, pair[0])
    output_tensor = tensorFromSentence(output_lang, pair[1])

    return (input_tensor, output_tensor)


if __name__ == '__main__':
    in_lang, out_lang, pairs = readLangs('eng', 'cha', reverse=False)
    print(len(pairs))
    print(pairs[10])
    print(tensorFromPair(pairs[10]))
    a = [tensorFromPair(random.choice(pairs)) for i in range(5)]
    print(a[4])
