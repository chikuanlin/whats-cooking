import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.path.pardir))
sys.path.append(PARENT_DIR)
sys.path.append(BASE_DIR)
from nn_solver import NNSolver
from processors.simple_ingredients_encoder import SimpleIngredientsEncoder
from dataset.dataset import WhatsCookingDataset
from dataset.dataset import WhatsCookingStemmedDataset
from dataset.dataset import WhatsCookingStemmedSeparatedDataset
import numpy as np
from sklearn.model_selection import KFold
from processors.tf_idf import TfIdf


if __name__ == "__main__":
    
    # Raw dataset
    # dataset = WhatsCookingDataset()
    
    # Stemmed dataset
    # dataset = WhatsCookingStemmedDataset()
    
    # Stemmed and ingredient separated dataset
    dataset = WhatsCookingStemmedSeparatedDataset()
    
    # Simple encoder
    # encoder = SimpleIngredientsEncoder()
    
    # TF-IDF
    encoder = TfIdf(1500)
    
    
    x = encoder.fit_transform(dataset)
    y = np.array([dataset.cuisine2id[cuisine.cuisine] for cuisine in dataset.cuisines])
    
    # kfold = KFold(n_splits=5, random_state=0, shuffle=True)

    # for i, (train_index, test_index) in enumerate(kfold.split(x)):
    #     print('Cross-Validation [%d/5]' % (i+1))
    #     x_train, x_test = x[train_index], x[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #     solver = NNSolver(dataset, in_features=x_train.shape[1])
    #     solver.train(x_train, y_train, x_test=x_test, y_test=y_test)

    solver = NNSolver(dataset, in_features=x.shape[1])
    solver.train(x, y)
        
    cuisines = dataset.load_test_file()
    x_test = encoder.transform(cuisines)
    solver.test(x_test, cuisines)