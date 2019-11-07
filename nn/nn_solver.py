import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.path.pardir))
sys.path.append(PARENT_DIR)
sys.path.append(BASE_DIR)
from base_solver import BaseSolver
from nn_model import WhatsCookingNet
from nn_dataset import NNDataset

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

class NNSolver(BaseSolver):
    
    def __init__(self, dataset, in_features=6714):
        super(NNSolver, self).__init__(dataset, in_features)

        # Neural network
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net = None
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None

        # default training parameters
        self.learning_rate = 1e-3
        self.batch_size = 128
        self.weight_decay = 1e-5
        self.step_size = 10
        self.gamma = 0.1
        self.epochs = 5

        # loss history
        self.train_loss_history = []
        self.test_loss_history = []
        self.train_acc_history = []
        self.test_acc_history = []

    def train(self, x_train, y_train, x_test=None, y_test=None, params=None):
        if params is not None:
            self._load_params(params)
        use_val = x_test is not None and y_test is not None
        
        self.net = WhatsCookingNet(in_features=self.in_features).to(self.device).double()
        self.optimizer = Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = StepLR(self.optimizer, self.step_size, gamma=self.gamma)

        train_dataset = NNDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        if use_val:
            test_dataset = NNDataset(x_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        for epoch in range(self.epochs):
            train_loss, train_acc = self._train_one_epoch(train_loader)
            if use_val:
                test_loss, test_acc = self._test_one_epoch(test_loader)
            else:
                test_loss, test_acc = 0.0, 0.0
            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)    
            self.test_loss_history.append(test_loss)
            self.test_acc_history.append(test_acc)            
            print('Epoch [%3d/%3d] train loss: %.6f train acc: %.4f val loss: %.6f val acc: %.4f' %(
                epoch+1, self.epochs, train_loss, train_acc, test_loss, test_acc))
        self._save_model()

    def test(self, x, cuisines):
        x = torch.from_numpy(x).to(self.device)
        self.net = self.net.eval()
        with torch.no_grad():
            output = self.net(x)
        pred_labels = torch.max(output, dim=1)[1]
        pred_labels = pred_labels.tolist()
        ids = [cuisine.id for cuisine in cuisines]
        pred_cuisines = [self.dataset.id2cuisine[label] for label in pred_labels]
        self._write2csv(ids, pred_cuisines)
    
    def _load_params(self, params):
        self.learning_rate = params.get('learning_rate', self.learning_rate)
        self.weight_decay = params.get('weight_decay', self.weight_decay)
        self.batch_size = params.get('batch_size', self.batch_size)
        self.step_size = params.get('step_size', self.step_size)
        self.gamma = params.get('gamma', self.gamma)
        self.epochs = params.get('epochs', self.epochs)

    def _train_one_epoch(self, train_loader):
        self.net = self.net.train()
        train_loss = 0.0
        train_acc = 0.0
        for x, labels in train_loader:
            x = x.to(self.device)
            labels = labels.to(self.device)
            output = self.net(x)
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            train_acc += self._check_accuracy(output, labels)
            self.optimizer.zero_grad()
        self.scheduler.step()
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        return train_loss, train_acc

    def _test_one_epoch(self, test_loader):
        self.net = self.net.eval()
        test_loss = 0.0
        test_acc = 0.0
        with torch.no_grad():
            for x, labels in test_loader:
                x = x.to(self.device)
                labels = labels.to(self.device)
                output = self.net(x)
                loss = self.criterion(output, labels)
                test_loss += loss.item()
                test_acc += self._check_accuracy(output, labels)
            test_loss /= len(test_loader)
            test_acc /= len(test_loader)
        return test_loss, test_acc

    @staticmethod
    def _check_accuracy(output, labels):
        return torch.mean((torch.max(output, dim=1)[1] == labels).float()).item()

    def _save_model(self):
        torch.save(self.net.state_dict(), self.model_name + '.pth')

    def load_model(self, model_path):
        self.net = WhatsCookingNet(in_features=self.in_features).to(self.device)
        self.net.load_state_dict(torch.load(model_path))