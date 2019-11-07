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