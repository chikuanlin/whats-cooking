import json
import os
import sys
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
# import re 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.path.pardir))
sys.path.append(PARENT_DIR)

class Cuisine:
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', None)
        self.cuisine = kwargs.get('cuisine', None)
        self.ingredients = kwargs.get('ingredients', [])
    
    def __str__(self):
        info = 'Id: {}\nCuisine: {}\nIngredients: '.format(self.id, self.cuisine)
        return info + ', '.join(self.ingredients)

class WhatsCookingDataset:
    def __init__(self, file_path='dataset/train.json'):
        self.cuisines = []
        self.id2ingredient = []
        self.ingredient2id = {}
        self.id2cuisine = []
        self.cuisine2id = {}
        with open(file_path, encoding='utf-8', mode = 'r') as json_file:
            for item in json.load(json_file):
                self.cuisines.append(Cuisine(**item))
                if item['cuisine'] not in self.cuisine2id:
                    self.id2cuisine.append(item['cuisine'])
                    self.cuisine2id[item['cuisine']] = len(self.id2cuisine)-1
                for ingredient in item['ingredients']:
                    if ingredient not in self.ingredient2id:
                        self.id2ingredient.append(ingredient)
                        self.ingredient2id[ingredient] = len(self.id2ingredient)-1

    def __len__(self):
        return len(self.cuisines)

    def __getitem__(self, index):
        return self.cuisines[index]

    def save_to_text(self, sorting=True):
        if sorting:
            cuisine_output = self.id2cuisine.copy()
            ingredient_output = self.id2ingredient.copy()
            cuisine_output.sort()
            ingredient_output.sort()
        else:
            cuisine_output = self.id2cuisine
            ingredient_output = self.id2ingredient

        with open('cuisine_labels.txt', 'w', encoding='utf-8') as f:
            for cuisine in cuisine_output:
                f.write(cuisine + '\n')
        with open('ingredients_labels.txt', 'w', encoding='utf-8') as f:
            for ingredient in ingredient_output:
                f.write(ingredient + '\n')

    def load_test_file(self, file_path='dataset/test.json'):
        cuisines = []
        with open(file_path, encoding='utf-8', mode = 'r') as json_file:
            for item in json.load(json_file):
                cuisines.append(Cuisine(**item))
        return cuisines
    
    
class WhatsCookingStemmedDataset:
    def __init__(self, file_path='dataset/train.json'):
        print("Loading and stemming What's Cooking dataset ...")
        self.cuisines = []
        self.id2ingredient = []
        self.ingredient2id = {}
        self.id2cuisine = []
        self.cuisine2id = {}
        self.wordcount = {} # used to calculate df in tf-idf
        porter = PorterStemmer()
        english_stopwords = set(stopwords.words('english'))
        punctuations = string.punctuation
        with open(file_path, encoding='utf-8', mode = 'r') as json_file:
            for item in json.load(json_file):
                self.cuisines.append(Cuisine(**item))
                if item['cuisine'] not in self.cuisine2id:
                    self.id2cuisine.append(item['cuisine'])
                    self.cuisine2id[item['cuisine']] = len(self.id2cuisine)-1
                for ingredient in item['ingredients']:
                    token_ingredient = word_tokenize(ingredient.lower())
                    stemmed_ingredient_tokens = [
                        # remove all parenthesis and words inside
                        # porter.stem(re.sub("[\(\[].*?[\)\]]", "", token)) 
                        porter.stem(token) 
                        for token in token_ingredient
                        if not token in punctuations
                        and not token in english_stopwords 
                        and len(token) > 1
                    ]
                    # count occurrence of tokens in all samples
                    for token in stemmed_ingredient_tokens:
                        try:
                            self.wordcount[token] += 1
                        except:
                            self.wordcount[token] = 1
                    ingredient = " ".join(stemmed_ingredient_tokens)
                    if ingredient not in self.ingredient2id:
                        self.id2ingredient.append(ingredient)
                        self.ingredient2id[ingredient] = len(self.id2ingredient)-1
        print("Successfully loaded stemmed What's Cooking dataset!")
        print(
            "# of cuisines = %d; # of ingredients = %d" \
            % (len(self.id2cuisine), len(self.id2ingredient)),
        )

    def __len__(self):
        return len(self.cuisines)

    def __getitem__(self, index):
        return self.cuisines[index]

    def save_to_text(self, sorting=True):
        if sorting:
            cuisine_output = self.id2cuisine.copy()
            ingredient_output = self.id2ingredient.copy()
            cuisine_output.sort()
            ingredient_output.sort()
        else:
            cuisine_output = self.id2cuisine
            ingredient_output = self.id2ingredient

        with open('cuisine_labels.txt', 'w', encoding='utf-8') as f:
            for cuisine in cuisine_output:
                f.write(cuisine + '\n')
        with open('ingredients_labels.txt', 'w', encoding='utf-8') as f:
            for ingredient in ingredient_output:
                f.write(ingredient + '\n')

    def load_test_file(self, file_path='dataset/test.json'):
        cuisines = []
        with open(file_path, encoding='utf-8', mode = 'r') as json_file:
            for item in json.load(json_file):
                cuisines.append(Cuisine(**item))
        return cuisines


if __name__ == "__main__":
    dataset = WhatsCookingDataset()
    print(dataset[0])
    dataset.save_to_text()
    test = dataset.load_test_file()
    print(test[0])