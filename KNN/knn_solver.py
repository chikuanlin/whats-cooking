import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.path.pardir))
sys.path.append(PARENT_DIR)
sys.path.append(BASE_DIR)
from base_solver import BaseSolver
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
#import torchvision.transforms as transforms
#import torchvision.datasets as datasets
#import torchvision.models as models
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, CosineAnnealingLR
from sklearn.metrics.pairwise import pairwise_distances
from metric_learn import LMNN

class knn_solver(BaseSolver):
    
    def __init__(self, dataset, in_features=6714):
        super().__init__(dataset, in_features)

    def PCA(self, x_train, x_test, k=500):
        print(x_train.shape)
        X_mean = x_train.mean(0, keepdim=True)
        X = x_train - X_mean
        X_te = x_test - X_mean

        U,S,V = torch.svd(torch.t(x_train.cuda()))
        return torch.mm(X,U[:,:k].cpu()), torch.mm(X_te,U[:,:k].cpu())


    def get_dist(self, x_train, x_test, metric="braycurtis"):
        print(x_train.shape[1])
        #metric="cosine"
        #x_train, x_test = self.PCA(x_train, x_test)
        n0 = x_train.shape[0]
        n1 = x_test.shape[0]
        dist = pairwise_distances(x_train.numpy(), x_test.numpy(), metric=metric)
        dist = torch.from_numpy(dist)
          
        return dist.float()


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
        x_train = np.array(x_train)
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
          x_train = torch.tensor(x_train).float()
          y_train = torch.tensor(y_train)
          x_train = x_train[shuffle_idx]
          y_train = y_train[shuffle_idx]
          x_val = x_train[35000:]
          x_tr  = x_train[:35000]
          y_val = y_train[35000:]
          y_tr  = y_train[:35000]
          
          use_DML = False
          
          if use_DML:
            x_val = x_train[5000:6000]
            x_tr  = x_train[:5000]
            y_val = y_train[5000:6000]
            y_tr  = y_train[:20000]
            x_tr, x_val = self.PCA(x_tr, x_val, 64)
            lmnn = LMNN(k=15, learn_rate=1e-6, min_iter=50, max_iter=100)
            lmnn.fit(x_tr.numpy(), y_tr.numpy())
            M = lmnn.get_mahalanobis_matrix()
            M = torch.tensor(M).float()
            n, d = x_val.shape
            m = x_tr.shape[0]
            x0 = x_tr.unsqueeze(1).expand(-1, n, -1).contiguous().view(-1,d)
            x1 = x_val.unsqueeze(0).expand(m,-1, -1).contiguous().view(-1,d)
            x = x0-x1
            dist0 = torch.mm(M, x.t().contiguous())
            dists = dist0.t().contiguous() * x
            dist = dists.sum(1).view(m,n)
          else:
            x_tr, x_val = self.PCA(x_tr, x_val, 500)
            dist = self.get_dist(x_tr, x_val).cpu()
        
          for k in [1,3,5,8,10,15,20,25,30]:
            y_pred = self.predict(dist, y_tr, k)
            acc = (y_pred == y_val).sum().float().numpy()/y_val.shape[0]
            print("K=",k,"  acc=",acc)
        torch.cuda.empty_cache()
        



class deep_similairy(nn.Module):
    def __init__(self, d=500):
        super().__init__()
        self.M = torch.nn.Parameter(torch.ones(d,d, dtype=torch.float32))
        self.net0 = nn.Sequential(
                      nn.Linear(d, 128),
                      nn.ReLU(),
                      nn.Linear(128, 1),
                      nn.ReLU()
                     )
    def forward(self, x0, x1):
        d  = x0.shape[1]
        xd = (x0-x1).abs()
        xs = (x0.abs()+x1.abs()).sum(1).view(-1,1)
        d1 = self.net0(xd)
        return d1/xs

class model(nn.Module):
    def __init__(self, d=500):
        super().__init__()
        self.sim = deep_similairy(d)
    
    def forward(self, x, x1=None):
        B = x.shape[0]
        D = x.shape[1]
        
        if x1 is not None:
          K = x1.shape[0]
          dist = torch.zeros(B,K,dtype=x.dtype,device=x.device)
          for i in range(B):
            if (i%3000) == 0:
              print(i)
            xa = x[i,:].unsqueeze(0).expand(K, -1).contiguous()
            xd = xa-x1
            dist[i,:] = self.sim(xa, x1).view(-1)
          return dist
        else:
          xa = x.clone().unsqueeze(0).expand(B, -1, -1).contiguous().view(-1,D)
          xb = x.clone().unsqueeze(1).expand(-1, B, -1).contiguous().view(-1,D)
          x0 = xb - xa
          x = self.sim(xa, xb).view(-1)
          return x


def weights_init(m):
   if type(m) in [nn.Linear]:
      nn.init.xavier_normal_(m.weight) 
      #m.weight.data.uniform_(-0.001,0.001)
      nn.init.zeros_(m.bias)

class Loader(torch.utils.data.Dataset):
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        
    def __len__(self):
        return len(self.x_train)
        
    def __getitem__(self, index):
        x = self.x_train[index]
        y = self.y_train[index]
        return x, y
        
        

class knn_dml_solver(BaseSolver):
    
    def __init__(self, dataset, in_features=6714):
        super().__init__(dataset, in_features)
        self.model = model().cuda()
        self.model.apply(weights_init)
        print(model)
    
    def PCA(self, x_train, x_test, k=500):
        print(x_train.shape)
        X_mean = x_train.mean(0, keepdim=True)
        X = x_train - X_mean
        X_te = x_test - X_mean
                                    
        U,S,V = torch.svd(torch.t(x_train.cuda()))
        return torch.mm(X,U[:,:k].cpu()), torch.mm(X_te,U[:,:k].cpu())
                                                    
    def test(self, x_train, y_train, x_test, cuisines=None):
        state = torch.load('knn_dml_best_model.pth')
        self.model.load_state_dict(state)
        
        x_train = torch.tensor(x_train, dtype=torch.float32)
        x_test  = torch.tensor(x_test, dtype=torch.float32)
        y_train = torch.tensor(y_train)
        x_train, x_test = self.PCA(x_train, x_test)
        
        dist = self.get_dist(x_train.cuda(), x_test.cuda()).cpu()
        y_pred = self.predict(dist, y_train, 20)
        ids = [cuisine.id for cuisine in cuisines]
        pred_cuisines = [self.dataset.id2cuisine[label] for label in y_pred]
        self._write2csv(ids, pred_cuisines)        
        

    def train(self, x_train, y_train):
        idx = np.random.permutation(len(x_train))
        print('samples:',len(x_train))
        x_train = np.array(x_train)[idx]
        y_train = np.array(y_train)[idx]
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train)
        x_val = x_train[32000:]
        x_tr  = x_train[:32000]
        y_val = y_train[32000:]
        y_tr  = y_train[:32000]
        #x_val = x_train[5000:6000]
        #y_val = y_train[5000:6000]
        #x_tr  = x_train[:5000]
        #y_tr  = y_train[:5000]
        y_tr_ = y_tr.clone()
        
        x_tr, x_val = self.PCA(x_tr, x_val)
        print('PCA done', x_tr.shape)    
        
        optimizer = Adam(self.model.parameters(), lr=0.0001, weight_decay=5e-3)
        scheduler = ExponentialLR(optimizer, 1)
        loss_fn = nn.MSELoss()

        train_loader = torch.utils.data.DataLoader(
            Loader(x_tr, y_tr), batch_size=256, shuffle=True)
            
        best_acc=0
        dist = self.get_dist(x_tr.cuda(), x_val.cuda()).cpu()
        for k in [1,3,5,8,10,15,20,25,30]:
          y_pred = self.predict(dist, y_tr_, k)
          acc = (y_pred == y_val).sum().float().numpy()/y_val.shape[0]
          print("K=",k,"  acc=",acc)
        
        for epoch in range(200):
          self.model.train()
          scheduler.step()
          loss_ = acc_ = cnt = yc = 0
          for i, (input,  target) in enumerate(train_loader):
            optimizer.zero_grad()

            B = target.shape[0]
            gt_p = target.clone().cuda().view(1,B).float()
            gt = target.clone().cuda()
            output = self.model(input.cuda())
            dists = output.view(B,B)
            dm = dists.sum(0).view(1,-1)
            #dists = dists / dm

            sorted, ind = dists.sort(dim=0, descending=False)
            sorted = sorted[:20]
            ind =ind[:20]

            y_p = gt[ind]

            gt_p = gt_p.expand(20,-1).contiguous().float()
            
            y_p = y_p.float() - gt_p
            y_p[y_p!=0] = 1
            yy = torch.sum(y_p)
            
            loss0 = torch.div(1,sorted[y_p!=0])
            
            loss1 = sorted[y_p==0]
            
            loss = loss0.mean() + loss1.mean()
            loss.backward()

            optimizer.step()
            lr = scheduler.get_lr()
            
            yc += yy.cpu().data.numpy()
            loss_ += loss.cpu().data.numpy()
            cnt += 1
            
          print('Epoch %2d: loss = %6.5f,  %5.3f, lr=%f'%(epoch,loss_/cnt, yc/cnt, lr[0]))
          loss_ = yc = 0
          
          if (epoch %20) ==19:
            dist = self.get_dist(x_tr.cuda(), x_val.cuda()).cpu()
            for k in [1,3,5,8,10,15,20,25,30]:
              y_pred = self.predict(dist, y_tr_, k)
              acc = (y_pred == y_val).sum().float().numpy()/y_val.shape[0]
              print("K=",k,"  acc=",acc)
              if k == 25:
                acc_25 =acc
            torch.save(self.model.state_dict(), 'knn_dml_checkpoint.pth')
            if best_acc <= acc_25:
              best_acc = acc_25
              torch.save(self.model.state_dict(), 'knn_dml_best_model.pth')


    def get_dist(self, x_train, x_test):

        n0 = x_train.shape[0]
        n1 = x_test.shape[0]
        torch.cuda.empty_cache()  
        with torch.no_grad():
          dist = self.model(x_train, x_test).view(n0,n1)
                                                            
        return dist
                                                                    
                                                                    
    def predict(self, dists, y_train, k=1):
        num_train, num_test = dists.shape
        num_class = y_train.max().data+1
        _, ind = dists.topk(k, dim=0, largest=False)
                                                                                  
        one_hot = torch.zeros(num_test, k, num_class, dtype=torch.int)
        idx0 = torch.arange(num_test).unsqueeze(1).expand(-1, k).contiguous().view(-1)
        idx1 = torch.arange(k).repeat(num_test)
        idx2 = y_train[ind.t().contiguous().view(-1)]

        one_hot[idx0, idx1, idx2] = 1
        one_hot = one_hot.sum(dim=1)

        v, ind = one_hot.topk(num_class)
        ind[ v < v[:,0:1] ] = num_class
        y_pred = ind.min(1)[0]
                
        return y_pred
        