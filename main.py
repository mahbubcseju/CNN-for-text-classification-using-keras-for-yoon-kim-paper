#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 11:33:25 2019

@author: mahbubcseju
"""

from data_load_and_preproccess import data_load_and_preproccess


word_size = 5000
sequence_length = 500

x_train, y_train, x_test, y_test, vocabolary_dict = data_load_and_preproccess(word_size,sequence_length)

