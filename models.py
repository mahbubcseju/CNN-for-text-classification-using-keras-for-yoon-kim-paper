#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 15:10:34 2019

@author: mahbubcseju
"""
from data_load_and_preproccess import data_load_and_preproccess
from word_to_vector_pretrained import word_to_vector
import numpy as np
from keras.layers import Input, Dropout, Convolution1D, MaxPooling1D, Flatten, Dense, Concatenate, Embedding
from keras.models import Model

class Models:
    def __init__(self, model_param):
        self.word_size = model_param['word_size']
        self.sequence_length = model_param['sequence_length']
        self.embedding_dim = model_param['embedding_dim']
        self.min_word_count = model_param['min_word_count']
        self.context_window_size = model_param['context_window_size']
        self.drop_prob = model_param['drop_prob']
        self.kernal_size = model_param['kernal_size']
        self.filters = model_param['num_filters']
        self.hidden_dims = model_param['hidden_dims']
        self.batch_size = model_param['batch_size']
        self.epochs = model_param['epochs']
#        print("Seriously man")

    def predict(self, x_text):
        predicted_result = self.model.predict(x_text)
        return predicted_result
        
        
    def cnn_static(self):
        x_train, y_train, x_test, y_test, vocabolary_dict = data_load_and_preproccess(self.word_size,
                                                                              self.sequence_length
                                                                              )
        embedding_weights = word_to_vector(np.vstack((x_train, x_test)), vocabolary_dict, self.embedding_dim, self.min_word_count,
                                       self.context_window_size)
        x_train = np.stack([np.stack([embedding_weights[word] for word in sentence ]) for  sentence in x_train])
        x_test = np.stack([np.stack([embedding_weights[word] for word in sentence ]) for  sentence in x_test])
        
        #model building
        model_input = Input(shape = (self.sequence_length, self.embedding_dim))
        model = Dropout(self.drop_prob[0])(model_input)
        
        multi_cnn_channel = []
        
        for kernal in self.kernal_size:
            conv_channel = Convolution1D(filters= self.filters,
                     kernel_size=kernal,
                     padding="valid",
                     activation="relu",
                     strides=1)(model)
            conv_channel = MaxPooling1D(pool_size = 2)(conv_channel)
            conv_channel = Flatten()(conv_channel)
            multi_cnn_channel.append(conv_channel)
        model = Concatenate()(multi_cnn_channel)
        model = Dropout(self.drop_prob[1])(model)
        
        for dimension in self.hidden_dims:
            model = Dense(dimension, activation = "relu")(model)
        model_output = Dense(1, activation = "sigmoid")(model)
        model = Model(model_input,model_output)
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        
        print("Started Training : ")
        model.fit(x_train, y_train, batch_size = self.batch_size, epochs= self.epochs, validation_data=(x_test, y_test), 
                  verbose=2)
        print("Training Completed")
        self.model = model
        
    def cnn_non_static(self):
            
        x_train, y_train, x_test, y_test, vocabolary_dict = data_load_and_preproccess(self.word_size,
                                                                              self.sequence_length
                                                                              )
        embedding_weights = word_to_vector(np.vstack((x_train, x_test)), vocabolary_dict, self.embedding_dim, self.min_word_count,
                                       self.context_window_size)
        embedding_weights = np.array([value for value in embedding_weights.values()])
        model_input = Input(shape = (self.sequence_length, ))
        model = Embedding(len(vocabolary_dict), self.embedding_dim, weights =[embedding_weights], input_length=self.sequence_length, name="embedding")(model_input)
        model = Dropout(self.drop_prob[0])(model)
        
        multi_cnn_channel = []
        
        for kernal in self.kernal_size:
            conv_channel = Convolution1D(filters= self.filters,
                     kernel_size=kernal,
                     padding="valid",
                     activation="relu",
                     strides=1)(model)
            conv_channel = MaxPooling1D(pool_size = 2)(conv_channel)
            conv_channel = Flatten()(conv_channel)
            multi_cnn_channel.append(conv_channel)
        model = Concatenate()(multi_cnn_channel)
        model = Dropout(self.drop_prob[1])(model)
        
        for dimension in self.hidden_dims:
            model = Dense(dimension, activation = "relu")(model)
        model_output = Dense(1, activation = "sigmoid")(model)
        model = Model(model_input,model_output)
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        
        print("Started Training : ")
        model.fit(x_train, y_train, batch_size = self.batch_size, epochs= self.epochs, validation_data=(x_test, y_test), 
                  verbose=2)
        print("Training Completed")
        self.model = model
    
            
            
            
            