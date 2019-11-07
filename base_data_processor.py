from sklearn.preprocessing import OneHotEncoder
import numpy as np

class BaseDataProcessor:
    
    def __init__(self):
        pass

    def fit(self, dataset):
        ''' Fit a model for data preprocessing
        param dataset: WhatsCookingDataset object
        '''
        raise NotImplementedError

    def transform(self, cuisines):
        ''' Transform data using current fitted parameters
        param cuisines: List of Cuisine
        return: 2D np array - num_cuisines x num_features
        '''
        raise NotImplementedError

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset.cuisines)

class SimpleIngredientsEncoder(BaseDataProcessor):
    def __init__(self):
        super(SimpleIngredientsEncoder, self).__init__()
        self.ingredient2id = {}

    def fit(self, dataset):
        print('fit data in dataset')
#        self.encoder.fit(np.array(dataset.id2ingredient).reshape((-1,1)))
        self.ingredient2id = dataset.ingredient2id

    def get_ingredient_index(self, ingredient):
        if (ingredient in self.ingredient2id):
            return self.ingredient2id[ingredient]
        else:
            return None
    
    def filter_none(self, list):
        return [item for item in list if item != None]
        
    def transform(self, cuisines):
        print('Use fitted data parameters to transform cuisines to feature vectors.')
#        test_ingredients = np.array(['garlic', 'seasoning']).reshape((-1, 1))
#        test_encoding = self.encoder.transform(test_ingredients).toarray() 
        encoded_ingredients = np.zeros((len(cuisines), len(self.ingredient2id)))
        for idx, cuisine in enumerate(cuisines):
            ingredients = cuisine.ingredients
            indices = self.filter_none(
                [self.get_ingredient_index(i) for i in ingredients],
            )
            encoded_ingredients[idx, indices] = 1
        return encoded_ingredients