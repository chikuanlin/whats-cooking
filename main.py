from dataset.dataset import WhatsCookingDataset
from dataset.dataset import WhatsCookingStemmedDataset
from dataset.dataset import WhatsCookingStemmedSeparatedDataset

from base_solver import BaseSolver
# from example_method_file.example_method import ExampleSolver
from SVM.SVC_solver import SVCSolver

from base_data_processor import BaseDataProcessor
from processors.simple_ingredients_encoder import SimpleIngredientsEncoder
from processors.tf_idf import TfIdf

if __name__ == "__main__":

    # dataset loading
    dataset = WhatsCookingDataset()
    train_y = [dataset.cuisine2id[cuisine.cuisine] for cuisine in dataset.cuisines]

    test_cuisines = dataset.load_test_file()
    
    # load stemmed dataset
    dataset_stemmed = WhatsCookingStemmedSeparatedDataset(stem=False)
    train_x_stemmed = dataset_stemmed.cuisines

    train_y_stemmed = [
        dataset_stemmed.cuisine2id[cuisine.cuisine] 
        for cuisine in dataset_stemmed.cuisines
    ]
    
    test_cuisines_stemmed = dataset_stemmed.load_test_file()

    # pre-processing
    processors = [processor for processor in BaseDataProcessor.__subclasses__()]

    p = processors[0]()
    train_x = p.fit_transform(dataset)
    test_x = p.transform(test_cuisines)

    tf_idf = TfIdf()
    train_x_tfidf = tf_idf.fit_transform(dataset_stemmed)
    test_x_tfidf = tf_idf.transform(test_cuisines_stemmed)

    # training and testing
    solvers = [SVCSolver]
    for solver in solvers:
        print('Now solving using {} solver'.format(solver.__name__))
        s = solver(dataset)
        s.train(train_x, train_y)
        s.test(test_x, test_cuisines)
