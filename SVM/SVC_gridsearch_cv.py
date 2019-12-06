import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.path.pardir))
sys.path.append(PARENT_DIR)
sys.path.append(BASE_DIR)
import numpy as np
from dataset.dataset import WhatsCookingStemmedSeparatedDataset
from processors.tf_idf import TfIdf
from processors.simple_ingredients_encoder import SimpleIngredientsEncoder
from SVM.SVC_solver import SVCSolver
from sklearn.model_selection import GridSearchCV


if __name__ == "__main__":
    
    # Stemmed and ingredient separated dataset
    dataset = WhatsCookingStemmedSeparatedDataset(stem=False)
    test_cuisines = dataset.load_test_file()
    
    # TF-IDF
    # encoder = TfIdf()
    encoder = SimpleIngredientsEncoder()
    x = encoder.fit_transform(dataset)
    test_x = encoder.transform(test_cuisines)
    y = np.array(
        [dataset.cuisine2id[cuisine.cuisine] for cuisine in dataset.cuisines])
    
    # Grid search with cross validation
    solver = SVCSolver(dataset, method='lsvc_ovr')
    parameters = {
        'estimator__penalty': ('l1', 'l2'), 
        'estimator__C': np.arange(0.2, 1.2, 0.2),
    }
    clf = GridSearchCV(solver.clf, parameters, cv=5)
    clf.fit(x, y)
    print(f'grid seach results: {clf.cv_results_}')
    print(f'best model parameters: {clf.best_params_}')
    print(f'best model score: {clf.best_score_}')

    # Save best model
    solver.clf = clf.best_estimator_
    solver._save_model()

    # Train best estimator on entire dataset
    solver = SVCSolver(dataset, method='lsvc_ovr')
    solver.train(x, y)
    solver.test(test_x, test_cuisines)
