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
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchwordemb
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, CosineAnnealingLR
from detect_ingrs import *

ingrW2VDim=512
irnnDim = 512
ingrW2V = 'vocab.bin'

class ingRNN(nn.Module):
    def __init__(self):
        super(ingRNN, self).__init__()
        self.irnn = nn.LSTM(input_size=ingrW2VDim, hidden_size=irnnDim, bidirectional=True, batch_first=True,  num_layers=2, dropout=0.2)
        _, vec = torchwordemb.load_word2vec_bin(ingrW2V)
        #self.embs = nn.Embedding(vec.size(0), ingrW2VDim, padding_idx=0) # not sure about the padding idx
        self.embs = nn.Embedding.from_pretrained(vec, freeze=True)
        #self.embs.weight.data.copy_(vec)
        
    def forward(self, x, sq_lengths):
        # we get the w2v for each element of the ingredient sequence
        x = self.embs(x)

        # sort sequence according to the length
        sorted_len, sorted_idx = sq_lengths.sort(0, descending=True)
        index_sorted_idx = sorted_idx\
                .view(-1,1,1).expand_as(x)
        sorted_inputs = x.gather(0, index_sorted_idx.long())
        # pack sequence
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(
        sorted_inputs, sorted_len.cpu().data.numpy(), batch_first=True)
        # pass it to the rnn
        out, hidden = self.irnn(packed_seq)
                
        # unsort the output
        _, original_idx = sorted_idx.sort(0, descending=False)
                
        # LSTM
        # bi-directional
        unsorted_idx = original_idx.view(1,-1,1).expand_as(hidden[0])
        # 2 directions x batch_size x num features, we transpose 1st and 2nd dimension
        output = hidden[0].gather(1,unsorted_idx).transpose(0,1).contiguous()
        output = output.view(output.size(0),output.size(1)*output.size(2))

        return output

class model(nn.Module):
    def __init__(self, classes=20):
        super().__init__()
        self.rnn = ingRNN()
        self.classify = nn.Sequential(
                          nn.Linear(irnnDim*4, classes),
                        )
    
    def forward(self, x, sq_len):
        x = self.rnn(x, sq_len)
        x = self.classify(x)
        return x

def weights_init(m):
   if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
     for name, param in m.named_parameters():
       if 'weight_ih' in name:
         torch.nn.init.xavier_uniform_(param.data)
       elif 'weight_hh' in name:
         torch.nn.init.orthogonal_(param.data)
       elif 'bias' in name:
         param.data.fill_(0)
         
   if type(m) in [nn.Linear]:
      nn.init.xavier_normal_(m.weight) 

class Loader(torch.utils.data.Dataset):
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.max_len = 150#max([len(ingr) for ingr in x_train])+1
        print('max length=',self.max_len)
        
    def __len__(self):
        return len(self.x_train)
        
    def __getitem__(self, index):
        np.random.shuffle(self.x_train[index])
        x = self.x_train[index]
        y = self.y_train[index]
        np.random.shuffle(x)
        x = detect_ingrs(x)
        s = len(x)
        x0 = torch.zeros(self.max_len, dtype=torch.long)
        x0[:s] = torch.Tensor(x)
        x = x0
        y = torch.Tensor([y]).long()
        return [x, s], y
        
        

class rnn_solver(BaseSolver):
    
    def __init__(self, dataset, in_features=6714):
        super().__init__(dataset, in_features)
        self.model = model().cuda()
        self.model.apply(weights_init)
        print(model)
    
    def test(self, x_test, cuisines=None):
        state = torch.load('rnn_best_model.pth')
        self.model.load_state_dict(state)
        y_test = torch.zeros(len(x_test))
        test_loader  = torch.utils.data.DataLoader(
            Loader(x_test, y_test), batch_size=256, shuffle=False)
        y_pred = []
        for i, (input,  target) in enumerate(test_loader):
            output = self.model(input[0].cuda(), input[1].cuda())
            y_pred_ = output.max(1)[1].cpu().numpy()
            for y in y_pred_:
              y_pred.append(y)
        ids = [cuisine.id for cuisine in cuisines]
        pred_cuisines = [self.dataset.id2cuisine[label] for label in y_pred]
        self._write2csv(ids, pred_cuisines)        
        

    def train(self, x_train, y_train):
        idx = np.random.permutation(len(x_train))
        print('samples:',len(x_train))
        x_train = np.array(x_train)[idx]
        y_train = np.array(y_train)[idx]
        x_val = x_train[35000:]
        x_tr  = x_train[:35000]
        y_val = y_train[35000:]
        y_tr  = y_train[:35000]
        print(np.max(y_tr))
        
        #optimizer = SGD(self.model.parameters(), lr=0.1, weight_decay=1e-4, nesterov=True, momentum=0.9)
        optimizer = Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = ExponentialLR(optimizer, 0.98)
        loss_fn = nn.CrossEntropyLoss()

        train_loader = torch.utils.data.DataLoader(
            Loader(x_tr, y_tr), batch_size=256, shuffle=True)
        val_loader  = torch.utils.data.DataLoader(
            Loader(x_val, y_val), batch_size=256, shuffle=False)
            
        best_acc=0
        
        for epoch in range(50):
          self.model.train()
          scheduler.step()
          loss_ = acc_ = cnt = 0
          for i, (input,  target) in enumerate(train_loader):
            output = self.model(input[0].cuda(), input[1].cuda())
            optimizer.zero_grad()
            loss = loss_fn(output, target.cuda().view(-1))
            loss.backward()
            optimizer.step()
            lr = scheduler.get_lr()
            
            pred = output.max(1)[1].cpu()
            match = torch.sum(pred == target.view(-1)).float()/target.shape[0]
            loss_ += loss.cpu().data.numpy()
            acc_  += match.data.numpy()
            cnt += 1
          print('Epoch %2d: loss = %6.5f, training acc=%5.3f, lr=%f'%(epoch,loss_/cnt, acc_*100/cnt, lr[0]))
          loss_ = acc_ = 0
              
          acc_ = 0            
          val_cnt = 0
          self.model.eval()
          for i, (input,  target) in enumerate(val_loader):
            with torch.no_grad():
              output = self.model(input[0].cuda(), input[1].cuda())
            pred = output.max(1)[1].cpu()
            acc_ += torch.sum(pred == target.view(-1)).float()/target.shape[0]
            val_cnt += 1
          star = '*' if best_acc <= (acc_/val_cnt) else ''
          print('val acc= %5.3f'%(acc_/val_cnt) + star)
          torch.save(self.model.state_dict(), 'rnn_checkpoint.pth')
          if best_acc <= (acc_/val_cnt):
            best_acc = (acc_/val_cnt)
            torch.save(self.model.state_dict(), 'rnn_best_model.pth')
            
            
           
          
            

