from dataset import WhatsCookingDataset

from base_solver import BaseSolver
from example_method_file.example_method import ExampleSolver

from base_data_processor import BaseDataProcessor
from base_data_processor import ExampleProcessor1

if __name__ == "__main__":
    # dataset loading
    dataset = WhatsCookingDataset()
    train_y = [dataset.cuisine2id[cuisine.cuisine] for cuisine in dataset.cuisines]

    test_cuisines = dataset.load_test_file()

    # pre-processing
    processors = [processor for processor in BaseDataProcessor.__subclasses__()]

    p = processors[0]()
    train_x = p.fit_transform(dataset)
    test_x = p.transform(test_cuisines)

    # training and testing
    solvers = [solver for solver in BaseSolver.__subclasses__()]
    for solver in solvers:
        print('Now solving using {} solver'.format(solver.__name__))
        s = solver()
        # s.train(train_x, train_y)
        # s.test_x(test_x)
