import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.path.pardir))
sys.path.append(PARENT_DIR)
sys.path.append(BASE_DIR)

from base_data_processor import BaseDataProcessor
import numpy as np


class SimpleIngredientsEncoder(BaseDataProcessor):
    def __init__(self):
        super(SimpleIngredientsEncoder, self).__init__()
        self.ingredient2id = {}

    def fit(self, dataset):
        self.ingredient2id = dataset.ingredient2id
        
    def transform(self, cuisines):
        encoded_ingredients = np.zeros((len(cuisines), len(self.ingredient2id)))
        for idx, cuisine in enumerate(cuisines):
            indices = [
                self.ingredient2id[ingredient] 
                for ingredient in cuisine.ingredients 
                if ingredient in self.ingredient2id
            ]            
            encoded_ingredients[idx, indices] = 1
        return encoded_ingredients