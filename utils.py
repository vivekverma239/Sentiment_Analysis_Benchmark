from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import csv
import tensorflow as tf
import pandas as pd
import re
import numpy as np
import pickle

class DataProcessor(object):
    def __init__(self,data_file,vocab_size=20000,seperator=',',remove_special=True,lower=True, max_seq_len=50,header=None,
                    reverse=False, raw_data=False):
        self.data_file = data_file
        self.vocab_size = vocab_size
        self.seperator = seperator
        self.max_seq_len = max_seq_len
        self.raw_data = raw_data
        self.lower = lower
        self.reverse = reverse
        self.remove_special = remove_special
        self._raw_data , self._raw_labels = self._load_data(self.data_file,header=header)
        self.label_to_id = self._build_vocab_label()
        self.labels = np.asarray([self.label_to_id[i] for i in self._raw_labels])
        if not self.raw_data :
            self.word_to_id =  self._build_vocab()
            self.data = np.asarray(self._text_to_word_ids(self._raw_data))



    def _load_data(self,filename,contains_label=True,header=None):
        df = pd.read_csv(filename,sep=self.seperator,header=header)
        column_names = df.columns.values
        data = df[column_names[0]].values.tolist()
        if contains_label:
            label = [i.strip().lower() for i in df[column_names[1]].values.tolist()]
            return data, label
        return data

    def _split_to_words(self,text):
        if self.remove_special:
            text = re.sub(r'[^0-9a-zA-Z\?\.\s]', ' ', text.lower() if self.lower \
                                                                else text)
        return re.split('\s+',text)

    def _build_vocab(self):
        data = []
        for text in self._raw_data:
            data.extend(self._split_to_words(text))
        counter = collections.Counter(data)
        count_pairs = sorted(counter.most_common(self.vocab_size), key=lambda x: (-x[1], x[0]))
        words, _ = list(zip(*count_pairs))
        word_to_id = dict(zip(words, range(1,len(words)+1)))
        return word_to_id

    def _build_vocab_label(self):
        counter = collections.Counter(self._raw_labels)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        labels, _ = list(zip(*count_pairs))
        label_to_id = dict(zip(labels, range(0,len(labels))))
        return label_to_id

    def _text_to_word_ids(self,text_list,reverse=False):

        text_ids = []
        for text_items in text_list:

            data = self._split_to_words(text_items)
            if self.reverse:
                data.reverse()
            # text_ids.append(([self.word_to_id[word] for word in data if word in self.word_to_id] + \
            #                            [0]*self.max_seq_len)[:self.max_seq_len])
            temp = ([self.word_to_id[word] for word in data if word in self.word_to_id] )
            if len(temp) < self.max_seq_len:
                temp =  ([0]*self.max_seq_len + temp)[-self.max_seq_len:]
            else:
                temp = temp[:self.max_seq_len]
            text_ids.append(temp)
        return text_ids


    def _convert_one_hot(self,data):
        pass

    def get_training_data(self,raw_text=False):
        if raw_text or self.raw_data:
            return self._raw_data, self.labels
        return self.data, self.labels

    def process_test_file(self,filename,contains_label=False,header=None):
        if contains_label:
            raw_test_data, raw_labels = self._load_data(filename,contains_label,header)
            test_data = raw_test_data
            if not self.raw_data:
                test_data = np.asarray(self._text_to_word_ids(raw_test_data))
            labels = np.asarray([self.label_to_id[i] for i in raw_labels])
            return test_data, labels
        else:
            raw_test_data, raw_labels = self._load_data(filename,contains_label,header)
            test_data = raw_test_data
            if not self.raw_data:
                test_data = np.asarray(self._text_to_word_ids(raw_test_data))
            labels = np.asarray([self.label_to_id[i] for i in raw_labels])
            return test_data

    def _load_glove(self,dim):
        """ Loads GloVe data.
        :param dim: word vector size (50, 100, 200)
        :return: GloVe word table
        """
        word2vec = {}
        print('Loading Glove Data.. Please Wait.. ')
        path = "data/glove/glove.6B." + str(dim) + 'd'
        if os.path.exists(path + '.cache'):
            with open(path + '.cache', 'rb') as cache_file:
                word2vec = pickle.load(cache_file)

        else:
            # Load n create cache
            with open(path + '.txt') as f:
                for line in f:
                    l = line.split()
                    word2vec[l[0]] = [float(x) for x in l[1:]]

            with open(path + '.cache', 'wb') as cache_file:
                pickle.dump(word2vec, cache_file)

        print("Loaded Glove data")
        return word2vec

    def get_embedding(self,dim):
        embedding = np.random.normal(loc=0.0, scale=0.1, size=[len(self.word_to_id)+1,dim])
        glove = self._load_glove(dim)
        for item in self.word_to_id:
            if item.lower() in glove:
                embedding[self.word_to_id[item]] = glove[item]
        return embedding



if __name__ == '__main__':
    data_path = 'data/custom/LabelledData.txt'
    processor = DataProcessor(data_path,seperator=',,,',max_seq_len=30)
    X, y = processor.get_training_data()
    print(X.shape, y.shape)
