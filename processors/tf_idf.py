# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:43:00 2019

@author: Xinyi
"""

import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.path.pardir))
sys.path.append(PARENT_DIR)
sys.path.append(BASE_DIR)

from base_data_processor import BaseDataProcessor


class TfIdf(BaseDataProcessor):
    def __init__(self):
        super(TfIdf, self).__init__()
        self.all_words = []
        self.wordcount = {}

    def fit(self, dataset):
        self.wordcount = dataset.wordcount
        self.all_words = [word for word in self.wordcount]
        
    def transform(self, cuisines):
        tf_idf = np.zeros((len(cuisines), len(self.all_words)))
        # TODO: implement tf-idf
        return tf_idf