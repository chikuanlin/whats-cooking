import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.path.pardir))
sys.path.append(PARENT_DIR)
sys.path.append(BASE_DIR)
from base_solver import BaseSolver
from nn_solver import NNSolver
from processors.simple_ingredients_encoder import SimpleIngredientsEncoder
from dataset.dataset import WhatsCookingDataset
import numpy as np
from sklearn.model_selection import KFold

if __name__ == "__main__":
    dataset = WhatsCookingDataset()
    
    encoder = SimpleIngredientsEncoder()

    x = encoder.fit_transform(dataset)
    y = np.array([dataset.cuisine2id[cuisine.cuisine] for cuisine in dataset.cuisines])

    kfold = KFold(n_splits=5, random_state=0, shuffle=True)

    for train_index, test_index in kfold.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        solver = NNSolver(dataset)
        solver.train(x_train, y_train, x_test=x_test, y_test=y_test)
        print(solver.train_loss_history[-1], solver.test_loss_history[-1])
        

    cuisines = dataset.load_test_file()
    x_test = encoder.transform(cuisines)

    solver.test(x_test, cuisines)