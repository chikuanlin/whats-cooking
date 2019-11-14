import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.path.pardir))
sys.path.append(PARENT_DIR)
sys.path.append(BASE_DIR)
from base_solver import BaseSolver
import numpy as np
import torch

class knn_solver(BaseSolver):
    
    def __init__(self, dataset, in_features=6714):
        super().__init__(dataset, in_features)

    def get_dist(self, x_train, x_test):
        x_train = x_train.cuda().half()
        x_test = x_test.cuda().half()
        num_train = x_train.shape[0]
        num_test  = x_test.shape[0]
        x_test = x_test.t()
        dist = (x_train * x_train).sum(1).view(-1,1) + \
               (x_test * x_test).sum(0).view(1,-1) - \
               x_train.mm(x_test)*2
        return dist.cpu().float()

    def predict(self, dists, y_train, k=1):
        num_train, num_test = dists.shape
        num_class = y_train.max()+1
        _, ind = dists.topk(k, dim=0, largest=False)
        
        one_hot = torch.zeros(num_test, k, num_class, dtype=torch.int)
        idx0 = torch.repeat_interleave(torch.arange(num_test), k)
        idx1 = torch.arange(k).repeat(num_test)
        idx2 = y_train[ind.t().contiguous().view(-1)]
        one_hot[idx0, idx1, idx2] = 1
        one_hot = one_hot.sum(dim=1) 
            
        v, ind = one_hot.topk(num_class)
        ind[ v < v[:,0:1] ] = num_class
        y_pred = ind.min(1)[0]
        
        return y_pred

    def train_test(self, x_train, y_train, x_test=None, cuisines=None, k=15):
        torch.cuda.empty_cache()
        testing = x_test is not None
        
        if testing:
          x_tr  = torch.tensor(x_train)
          y_tr  = torch.tensor(y_train)
          x_val = torch.tensor(x_test)
          dist = self.get_dist(x_tr, x_val)
          y_pred = self.predict(dist, y_tr, k)
          ids = [cuisine.id for cuisine in cuisines]
          pred_cuisines = [self.dataset.id2cuisine[label] for label in y_pred]
          self._write2csv(ids, pred_cuisines)
        else:
          shuffle_idx = torch.randperm(x_train.shape[0])
          x_train = torch.tensor(x_train)
          y_train = torch.tensor(y_train)
          x_train = x_train[shuffle_idx]
          y_train = y_train[shuffle_idx]
          x_val = x_train[35000:]
          x_tr  = x_train[:35000]
          y_val = y_train[35000:]
          y_tr  = y_train[:35000]
        
          dist = self.get_dist(x_tr, x_val)
        
          for k in [1,3,5,8,10,15,20,25,30]:
            y_pred = self.predict(dist, y_tr, k)
            acc = (y_pred == y_val).sum().float().numpy()/y_val.shape[0]
            print("K=",k,"  acc=",acc)
        torch.cuda.empty_cache()
        
          

