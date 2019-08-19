#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 11:42:49 2019

@author: mahbubcseju
"""

import numpy as np
from gensim.models import word2vec
from os.path import exists, split
import os


def word_to_vector(sentences, vocabolary_with_index, embedding_dim, min_word_count, context_window_size):
    
    #check already trined model is there or not
    pretrained_model_path = 'models/pretrained_word_vector'
    if exists(pretrained_model_path):
        print('Loading pretrained embedding model \'%s\'' % split(pretrained_model_path)[-1])
        embedding_model =word2vec.Word2Vec.load(pretrained_model_path) 
        print('Loaded pretrained model!!')
    else:
        print('Training word to vector pretrained model')
        number_of_thread = 2
        sampling_probability = .001 #Probability to remove some words
        sentences = [ [ vocabolary_with_index[w] for w in s] for s in sentences]
        embedding_model = word2vec.Word2Vec(sentences, workers = number_of_thread,
                                            size = embedding_dim, min_count = min_word_count, 
                                            window = context_window_size,
                                            sample = sampling_probability)
        embedding_model.init_sims(replace = True)
        
        if not exists('models'):
            os.mkdir('models')
        
        embedding_model.save(pretrained_model_path)
        print('Training Completed')

    embedding_weights = {key : embedding_model[word] if word in embedding_model
                         else np.random.uniform(-0.25, 0.25, embedding_model.vector_size)
                         for key, word in vocabolary_with_index.items()}
    return embedding_weights

#if __name__ == '__main__':
#    word_to_vector([],[])
    