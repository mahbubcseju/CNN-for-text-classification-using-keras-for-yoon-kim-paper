#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 11:04:41 2019

@author: mahbubcseju
"""
import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence

def data_load_and_preproccess(word_size,sequence_length):
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=word_size, start_char=None,
                                                              oov_char=None, index_from=None)
    np.load = np_load_old
    #Padding upto sequence length
    x_train = sequence.pad_sequences(x_train, maxlen=sequence_length, padding="post", truncating="post")
    x_test = sequence.pad_sequences(x_test, maxlen=sequence_length, padding="post", truncating="post")
    #Dictionary of vocabolary (word,int encoding value)
    vocabolary_dict = imdb.get_word_index()
    #Dictionary of vocabolary (int encoding value, word)
    vocabolary_dict_encode_first = dict((v,k) for k,v in vocabolary_dict.items())
    vocabolary_dict_encode_first[0] = '<PAD>'
    return x_train, y_train, x_test, y_test,vocabolary_dict_encode_first

 

if __name__ == '__main__':
    x_train, y_train, x_test, y_test, vocabolary_dict = data_load_and_preproccess(5000,500)
#    counter = 0
#    for k, v in vocabolary_dict.items():
#        counter = counter + 1
#        print(k, " ", v)
    print(x_train.shape, " ", y_train.shape, " ", len(vocabolary_dict))
    