import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.path.pardir))
sys.path.append(PARENT_DIR)
sys.path.append(BASE_DIR)
from dataset.dataset import WhatsCookingDataset, \
    WhatsCookingStemmedDataset, \
    WhatsCookingStemmedSeparatedDataset
import numpy as np
from sklearn.model_selection import KFold
from processors.tf_idf import TfIdf
from SVM.SVC_solver import SVCSolver
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

TF_IDF_K = 3010

if __name__ == "__main__":
    
    # Stemmed and ingredient separated dataset
    dataset = WhatsCookingStemmedSeparatedDataset()
    test_cuisines = dataset.load_test_file()
    
    # TF-IDF
    encoder = TfIdf(TF_IDF_K)
    
    x = encoder.fit_transform(dataset)
    y = np.array([dataset.cuisine2id[cuisine.cuisine] for cuisine in dataset.cuisines])
    test_x = encoder.transform(test_cuisines)
    

    kfold = KFold(n_splits=5, random_state=0, shuffle=True)

    solver = SVCSolver(dataset, in_features=TF_IDF_K)
    # parameters = {'estimator__penalty': ('l1', 'l2'), 'estimator__C': np.arange(0.2, 1.0, 0.2)}
    # clf = GridSearchCV(solver.clf, parameters, cv=5)
    # clf.fit(x, y)
    # solver.clf = clf.best_estimator_
    # print(clf.cv_results_)
    # print(clf.best_params_, clf.best_score_)

    print(cross_val_score(solver.clf, x, y, cv=5))


    # Refit best estimator on whole dataset
    solver.train(x, y)
    solver.test(test_x, test_cuisines)
