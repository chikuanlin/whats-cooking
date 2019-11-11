import os
import sys
import time
import numpy as np
from datetime import datetime
from base_solver import BaseSolver
import xgboost as xgb
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

class XGBSolver(BaseSolver):

    def __init__(self, dataset, in_features=6714):
        super(XGBSolver, self).__init__(dataset, in_features=in_features)
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.method = 'gbtree'
        self.model_name = f'{self.timestamp}_{self.method}'

        self.bst = xgb.XGBClassifier(
            max_depth=25,
            learning_rate=0.3,
            n_estimators=100,
            verbosity=1,
            objective='multi:softmax',
            tree_method='gpu_hist',
            reg_lambda=1
            # reg_alpha=1
        )

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
        evallist = [(evalX, evalY)]
        self.bst.fit(
            trainX, trainY,
            eval_set=evallist,
            early_stopping_rounds=5
        )
        print(f'training time: {time.time() - start_time} seconds')
        self._save_model()

    def test(self, x, cuisines):
        print('Testing started...')
        ypred = self.bst.predict(x, ntree_limit=self.bst.best_ntree_limit)
        ids = [cuisine.id for cuisine in cuisines]
        pred_cuisines = [self.dataset.id2cuisine[label] for label in ypred.astype(int)]
        self._write2csv(ids, pred_cuisines)

    def _save_model(self):
        print('Saving model...')
        pickle.dump(self.bst, open(os.path.join(BASE_DIR, self.model_name + '.model'), 'wb'))

    def load_model(self, model_path):
        print('Loading model...')
        self.bst = pickle.load(open(model_path, 'rb'))
    