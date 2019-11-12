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
    def __init__(self, k):
        super(TfIdf, self).__init__()
        # processor will select the top-k important ingredients
        self.k = k  
        self.ingredient2id = {}
        self.id2ingredient = []

    def fit(self, dataset): 
        print("Computing the %d most important words with tf-idf ..." % self.k)
        tf_idf_score = np.zeros((len(dataset.cuisines), len(dataset.ingredient2id)))
        cuisine2ingredientscount = {}
        ingredient2cuisine = {}
        N = len(cuisine2ingredientscount)
        # count the occurrence of each ingredient in each cuisine type
        # and the number of cuisine types that contains each ingredient
        for cuisine in dataset.cuisines:
            if (cuisine.cuisine not in cuisine2ingredientscount):
                cuisine2ingredientscount[cuisine.cuisine] = {}
            for ingredient in cuisine.ingredients:
                try:
                    cuisine2ingredientscount[cuisine.cuisine][ingredient] += 1
                    ingredient2cuisine[ingredient].add(cuisine.cuisine)
                except:
                    cuisine2ingredientscount[cuisine.cuisine][ingredient] = 1
                    ingredient2cuisine[ingredient] = {cuisine.cuisine}
        # calculate tf-idf score
        for cuisine_type, ingredients in enumerate(cuisine2ingredientscount):
            for ingredient, num_ingredient in enumerate(ingredients):
                if (ingredient not in dataset.ingredient2id):
                    continue
                ingredient_idx = dataset.ingredient2id[ingredient]
                cuisine_idx = dataset.cuisine2id[cuisine_type]
                tf_idf_score[cuisine_idx, ingredient_idx] = \
                    num_ingredient / len(ingredients) * \
                    np.log(N / (len(ingredient2cuisine[ingredient]) + 1))
        tf_idf_mean = self._compute_column_mean(tf_idf_score)
        top_k_indices = np.flip(np.argsort(tf_idf_mean))[0:self.k]
        self.id2ingredient = [dataset.id2ingredient[i] for i in top_k_indices]
        for ingredient_idx, ingredient in enumerate(self.id2ingredient):
            self.ingredient2id[ingredient] = ingredient_idx
        
    def transform(self, cuisines):
        print("Transforming input data with tf-idf ...")
        encoded_ingredients = np.zeros((len(cuisines), len(self.ingredient2id)))
        for idx, cuisine in enumerate(cuisines):
            indices = [
                self.ingredient2id[ingredient] 
                for ingredient in cuisine.ingredients 
                if ingredient in self.ingredient2id
            ]            
            encoded_ingredients[idx, indices] = 1
        return encoded_ingredients
    
    def _compute_column_mean(self, matrix):
        mean = np.zeros((matrix.shape[1]))
        for i in range(0, matrix.shape[1]):
            if (np.sum(matrix[:, i]) == 0):
                mean[i] = 0
            else:
                mean[i] = np.sum(matrix[:, i]) / np.count_nonzero(matrix[:, i])
        return mean