from dataset.dataset import WhatsCookingDataset
from dataset.dataset import WhatsCookingStemmedDataset

from base_solver import BaseSolver
from example_method_file.example_method import ExampleSolver

from base_data_processor import BaseDataProcessor
from processors.simple_ingredients_encoder import SimpleIngredientsEncoder

if __name__ == "__main__":
    # dataset loading
    dataset = WhatsCookingDataset()
    train_y = [dataset.cuisine2id[cuisine.cuisine] for cuisine in dataset.cuisines]
    id2ingredient = dataset.id2ingredient

    test_cuisines = dataset.load_test_file()
    
    # load stemmed dataset
    dataset_stemmed = WhatsCookingStemmedDataset()
    id2ingredient_stemmed = dataset_stemmed.id2ingredient
    id2cuisine_stemmed = dataset_stemmed.id2cuisine
    word_count = dataset_stemmed.wordcount
    
    # pre-processing
    processors = [processor for processor in BaseDataProcessor.__subclasses__()]
    
    p = processors[0]()
    train_x = p.fit_transform(dataset)
    test_x = p.transform(test_cuisines)

    # training and testing
    solvers = [solver for solver in BaseSolver.__subclasses__()]
    for solver in solvers:
        print('Now solving using {} solver'.format(solver.__name__))
        s = solver(dataset)
        # s.train(train_x, train_y)
        # s.test_x(test_x)
