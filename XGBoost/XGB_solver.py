import os
import sys
import time
import numpy as np
from datetime import datetime
from base_solver import BaseSolver
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

class XGBSolver(BaseSolver):

    def __init__(self, dataset, in_features=6714):
        super(XGBSolver, self).__init__(dataset, in_features=in_features)
        self.bst = xgb.Booster()
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.method = 'gbtree'
        self.model_name = f'{self.timestamp}_{self.method}'
        self.param = {
            # General Parameters
            'booster' : self.method,
            'verbosity' : 1,
            # Booster parameters
            # 'subsample': 0.8,
            # 'colsample_bynode': 0.8,
            # 'num_parallel_tree': 10,
            'learning_rate': 0.3,
            'max_depth': 25,
            'lambda': 1,
            # 'alpha': 1,
            'tree_method': 'gpu_hist',
            # Learning Task Parameters
            'objective': 'multi:softmax',
            'num_class': 20,
        }

    def train(self, x, y):
        def SplitData(x, y, ratio=0.1):
            y = np.array(y)
            n = x.shape[0]
            nEval = int(ratio * n)
            nTrain = n - nEval
            idxEval = np.random.choice(n, nEval, replace=False)
            idxTrain = np.array([i for i in range(n) if i not in idxEval])
            trainX = x[idxTrain, :]
            trainY = y[idxTrain]
            evalX = x[idxEval, :]
            evalY = y[idxEval]
            print(f'train set: {nTrain}, eval set: {nEval}')
            return trainX, trainY, evalX, evalY

        print('Training started...')
        start_time = time.time()
        trainX, trainY, evalX, evalY = SplitData(x, y, ratio=0.1)
        dtrain = xgb.DMatrix(data=trainX, label=trainY)
        deval = xgb.DMatrix(data=evalX, label=evalY)
        evallist = [(deval, 'eval')]
        self.bst = xgb.train(
            params=self.param,
            dtrain=dtrain,
            num_boost_round=200,
            evals=evallist,
            early_stopping_rounds=5
        )
        print(f'training time: {time.time() - start_time} seconds')
        self._save_model()

    def test(self, x, cuisines):
        print('Testing started...')
        dtest = xgb.DMatrix(data=x)
        ypred = self.bst.predict(dtest, ntree_limit=self.bst.best_ntree_limit)
        ids = [cuisine.id for cuisine in cuisines]
        pred_cuisines = [self.dataset.id2cuisine[label] for label in ypred.astype(int)]
        self._write2csv(ids, pred_cuisines)

    def _save_model(self):
        print('Saving model...')
        self.bst.save_model(os.path.join(BASE_DIR, self.model_name + '.model'))
        self.bst.dump_model(os.path.join(BASE_DIR, self.model_name + '.raw.txt'))

    def load_model(self, model_path):
        print('Loading model...')
        self.bst.load_model(model_path)
    