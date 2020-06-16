from nltk.tokenize import word_tokenize
import re
import collections
import pickle
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
import os
train_article_path = "sumdata/train/train.article.txt"
train_title_path = "sumdata/train/train.title.txt"
valid_article_path = "sumdata/train/valid.article.filter.txt"
valid_title_path = "sumdata/train/valid.title.filter.txt"
our_test_path = "NewsContents/AFRO2020050600002.txt"
our_test_trainingPath = "NewsContents/"

def clean_str(sentence):
    sentence = re.sub("[#.]+", "#", sentence)
    return sentence


def clean_title(file_name):
    file = open(file_name, "r", encoding='utf-8')
    filedata = file.readlines()
    article = []
    title = ''
    for eachLine in filedata:
        eachLine = eachLine[:-1]
        if len(eachLine) >= 1:
            if eachLine[:3] != 'URL' and eachLine[:2] != 'ID' and eachLine[:4] != 'Date' and eachLine[:8] != 'Ariticle':
                if eachLine[:5] == 'Title':
                    title += eachLine[8:]
                    continue
                if len(eachLine.split(" "))<5:
                    continue
                else:
                    article += eachLine.split("\n")
    return article, title


def clean_title_forTraining(dir_name):
    files = os.listdir(dir_name)
    article = []
    title = []
    for eachFile in files:
        file = open(dir_name+eachFile, "r", encoding='utf-8')
        filedata = file.readlines()
        a = " "
        for eachLine in filedata:
            eachLine = eachLine[:-1]
            if len(eachLine) >= 1:
                if eachLine[:3] != 'URL' and eachLine[:2] != 'ID' and eachLine[:4] != 'Date' and eachLine[:8] != 'Ariticle':
                    if eachLine[:5] == 'Title':
                        title.append(eachLine[8:])
                        continue
                    if len(eachLine.split(" "))<5:
                        continue
                    else:
                        article += eachLine.split("\n")
    return article, title


def get_text_list(data_path, toy):
    with open(data_path, "r", encoding="utf-8") as f:
        if not toy:
            return [clean_str(x.strip()) for x in f.readlines()]
        else:
            return [clean_str(x.strip()) for x in f.readlines()][:50000]


def build_dict(step, toy=False):
    if step == "train":
        #train_article_list = get_text_list(train_article_path, toy)
        #train_title_list = get_text_list(train_title_path, toy)
        train_article_list, train_title_list = clean_title_forTraining(our_test_trainingPath)
        words = list()
        for sentence in train_article_list:
            for word in word_tokenize(sentence):
                words.append(word)

        word_counter = collections.Counter(words).most_common()
        word_dict = dict()
        word_dict["<padding>"] = 0
        word_dict["<unk>"] = 1
        word_dict["<s>"] = 2
        word_dict["</s>"] = 3
        for word, _ in word_counter:
            word_dict[word] = len(word_dict)

        with open("word_dict.pickle", "wb") as f:
            pickle.dump(word_dict, f)

    elif step == "valid":
        with open("word_dict.pickle", "rb") as f:
            word_dict = pickle.load(f)

    reversed_dict = dict(zip(word_dict.values(), word_dict.keys()))

    article_max_len = 200
    summary_max_len = 100

    return word_dict, reversed_dict, article_max_len, summary_max_len


def build_dataset(step, word_dict, article_max_len, summary_max_len, toy=False):
    if step == "train":
        article_list = get_text_list(train_article_path, toy)
        title_list = get_text_list(train_title_path, toy)
    elif step == "train1":
        article_list,title_list = clean_title_forTraining(our_test_trainingPath)
    elif step == "valid":
        article_list = get_text_list(our_test_path, toy)
    elif step == "test":
        article_list,title_list = clean_title(toy)
    else:
        raise NotImplementedError

    x = [word_tokenize(d) for d in article_list]
    x = [[word_dict.get(w, word_dict["<unk>"]) for w in d] for d in x]
    x = [d[:article_max_len] for d in x]
    x = [d + (article_max_len - len(d)) * [word_dict["<padding>"]] for d in x]

    if step == "valid" or step =='test':
        return x

    else:
        y = [word_tokenize(d) for d in title_list]
        y = [[word_dict.get(w, word_dict["<unk>"]) for w in d] for d in y]
        y = [d[:(summary_max_len - 1)] for d in y]
        return x, y


# 타이틀 뺴낼것
# test로 해서 코드 짤것
#

def batch_iter(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]


def get_init_embedding(reversed_dict, embedding_size):
    glove_file = "glove/glove.42B.300d.txt"
    word2vec_file = get_tmpfile("word2vec_format.vec")
    glove2word2vec(glove_file, word2vec_file)
    print("Loading Glove vectors...")
    word_vectors = KeyedVectors.load_word2vec_format(word2vec_file)

    word_vec_list = list()
    for _, word in sorted(reversed_dict.items()):
        try:
            word_vec = word_vectors.word_vec(word)
        except KeyError:
            word_vec = np.zeros([embedding_size], dtype=np.float32)

        word_vec_list.append(word_vec)

    # Assign random vector to <s>, </s> token
    word_vec_list[2] = np.random.normal(0, 1, embedding_size)
    word_vec_list[3] = np.random.normal(0, 1, embedding_size)

    return np.array(word_vec_list)
