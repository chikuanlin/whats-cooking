import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.path.pardir))
sys.path.append(PARENT_DIR)
sys.path.append(BASE_DIR)
from base_solver import BaseSolver
from constants import *

class ExampleSolver(BaseSolver):
    
    def __init__(self, *args):
        super(ExampleSolver, self).__init__(*args)