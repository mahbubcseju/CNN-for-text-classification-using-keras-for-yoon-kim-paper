#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 11:33:25 2019

@author: mahbubcseju
"""
import sys
from data_load_and_preproccess import data_load_and_preproccess
from word_to_vector_pretrained import word_to_vector
import numpy as np
from models import Models

model_properties = {
        "word_size": 5000,
        "sequence_length": 500,
        "embedding_dim": 50,
        "context_window_size": 10,
        "min_word_count": 1,
        "drop_prob": ( 0.5, 0.8 ),
        "kernal_size" : (3, 5),
        "num_filters" : 10,
        "hidden_dims" : [50],
        "batch_size" : 64,
        "epochs" : 10,
        }

model = Models(model_properties)
model.cnn_non_static()

from data_load_and_preproccess import data_load_and_preprocess_csv
x_data, y_data = 
print(model.predict("data/test.csv",500))
#if(len(sys.argv)==2):
#    action = sys.argv[1]
#    model = Model(model_properties)
#    if action == 'CNN-static':
#        model.cnn_static()
#    else:
#        model.cnn_non_static()
#else:
#    print('please run like this : python main.py [model_name]')
#    print('please select model_name among [CNN-rand, CNN-static, CNN-non-static ]')
    



