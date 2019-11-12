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
        self.ingredient_count = {} # used to calculate df in tf-idf
        self.porter = PorterStemmer()
        self.english_stopwords = set(stopwords.words('english'))
        with open(file_path, encoding='utf-8', mode = 'r') as json_file:
            for item in json.load(json_file):
                if item['cuisine'] not in self.cuisine2id:
                    self.id2cuisine.append(item['cuisine'])
                    self.cuisine2id[item['cuisine']] = len(self.id2cuisine)-1
                ingredients_stemmed = []
                for ingredient in item['ingredients']:
                    ingredient_stemmed = self._stem_ingredient(ingredient)
                    if (len(ingredient_stemmed) > 1):
                        ingredients_stemmed.append(ingredient_stemmed)
                        try:
                            self.ingredient_count[ingredient_stemmed] += 1
                        except:
                            self.ingredient_count[ingredient_stemmed] = 1
                        if ingredient_stemmed not in self.ingredient2id:
                            self.id2ingredient.append(ingredient_stemmed)
                            self.ingredient2id[ingredient_stemmed] = \
                                len(self.id2ingredient) - 1
                self.cuisines.append(
                    Cuisine(**{
                        'id': item.get('id', None),
                        'cuisine': item['cuisine'],
                        'ingredients': ingredients_stemmed,
                    })
                )
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
                ingredients_stemmed = [
                    self._stem_ingredient(ingredient)
                    for ingredient in item['ingredients']
                ]
                cuisines.append(
                    Cuisine(**{
                        'id': item.get('id', None),
                        'cuisine': item.get('cuisine', None),
                        'ingredients': ingredients_stemmed,
                    })
                )
        return cuisines
    
    def _stem_ingredient(self, ingredient):
        token_ingredient = word_tokenize(ingredient.lower())
        token_ingredient_rm_punc = [
            self._remove_punctuation(token)
            for token in token_ingredient
        ]
        stemmed_ingredient_tokens = [
            # remove all parenthesis and words inside
            # porter.stem(re.sub("[\(\[].*?[\)\]]", "", token)) 
            self.porter.stem(token)
            for token in token_ingredient_rm_punc
            if not token in self.english_stopwords
        ]
        ingredient_stemmed = " ".join(stemmed_ingredient_tokens)
        return ingredient_stemmed
    
    def _remove_punctuation(self, token):
        for punctuation in string.punctuation:
            token = token.replace(punctuation, '')
        return token


if __name__ == "__main__":
    dataset = WhatsCookingDataset()
    print(dataset[0])
    dataset.save_to_text()
    test = dataset.load_test_file()
    print(test[0])