import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.path.pardir))
sys.path.append(PARENT_DIR)
sys.path.append(BASE_DIR)
from base_solver import BaseSolver
import numpy as np
import torch

class bayes_solver(BaseSolver):
    
    def __init__(self, dataset, in_features=6714):
        super().__init__(dataset, in_features)
        self.theta = None
        self.pi = None

    def train(self, x_train, y_train, alpha=1, beta=1):
        if not isinstance(x_train, torch.Tensor):
          x_train = torch.tensor(x_train)
          y_train = torch.tensor(y_train)
        n,d = x_train.shape
        c   = y_train.max()+1
        theta = torch.zeros(d,c).float()
        pi = torch.zeros(c)
        for i in range(c):
          pi[i] = ((y_train == i).sum().float() +alpha) / (n+alpha+beta)
          idx = (y_train == i)
          for j in range(d):
            theta[j,i] = (x_train[idx,j].sum().float() + alpha) / (idx.sum()+alpha+beta) 
            
        self.pi      = torch.log(pi)
        self.theta   = torch.log(theta)
        self.theta_b = torch.log(1-theta)
        self.num_class = c
        return pi, theta
    
    def predict(self, x_test):
        if not isinstance(x_test, torch.Tensor):
          x_test = torch.tensor(x_test)
        n,d = x_test.shape
        c = self.num_class
        pc = torch.zeros(n,c)
        for j in range(c):
          pc[:,j] = self.pi[j]
          for k in range(d):
            t  = (x_test[:, k] != 0).float()
            tb = (x_test[:, k] == 0).float()
            pc[:,j] += self.theta[k, j]*t + self.theta_b[k, j]*tb
        y_pred = torch.argmax(pc, 1)
        return y_pred
    
    def test(self, x, cuisines):
        if self.theta is None:
          print("error: bayes has not been trained")
        else:
          y_pred = self.predict(x)
          ids = [cuisine.id for cuisine in cuisines]
          pred_cuisines = [self.dataset.id2cuisine[label] for label in y_pred]
          self._write2csv(ids, pred_cuisines)

    def train_test(self, x_train, y_train):
        if not isinstance(x_train, torch.Tensor):
          x_train = torch.tensor(x_train)
          y_train = torch.tensor(y_train)
        shuffle_idx = torch.randperm(x_train.shape[0])
        x_train = x_train[shuffle_idx]
        y_train = y_train[shuffle_idx]
        x_val = x_train[35000:]
        x_tr  = x_train[:35000]
        y_val = y_train[35000:]
        y_tr  = y_train[:35000]
        
        
        self.train(x_tr, y_tr, 1, 1)
        y_pred = self.predict(x_val)
        acc = (y_pred == y_val).sum().float()/x_val.shape[0]
        print("Validation Accuracy=",acc)
        
        self.train(x_tr, y_tr, 1.5, 1.5)
        y_pred = self.predict(x_val)
        acc = (y_pred == y_val).sum().float()/x_val.shape[0]
        print("Validation Accuracy=",acc)
        
        self.train(x_tr, y_tr, 2, 2)
        y_pred = self.predict(x_val)
        acc = (y_pred == y_val).sum().float()/x_val.shape[0]
        print("Validation Accuracy=",acc)
                
          
        
        
          

