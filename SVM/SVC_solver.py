import os
import sys
import time
from base_solver import BaseSolver
import numpy as np
from datetime import datetime
from sklearn import svm
from sklearn import multiclass
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

class SVCSolver(BaseSolver):
    
    def __init__(self, *args, method='lsvc_ovr'):
        super(SVCSolver, self).__init__(*args)

        self.method = method
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.model_name = f'{self.timestamp}_{self.method}'

        # Init classifiers
        lsvc = svm.LinearSVC(
            verbose=1,
            dual=False,
            # loss='hinge',
            penalty='l1',
            C=0.5
        )
        lsvc_ovr = multiclass.OneVsRestClassifier(lsvc, n_jobs=-1)
        classifiers = {
            'lsvc': lsvc,
            'lsvc_ovr': lsvc_ovr,
        }

        self.clf = classifiers[self.method]

    def train(self, x, y):
        print('Training started...')
        start_time = time.time()
        self.clf.fit(x, y)
        print(f'training time: {time.time() - start_time} seconds')
        print(f'training score: {self.clf.score(x, y)}')
        self._save_model()

    def test(self, x, cuisines):
        print('Testing started...')
        ypred = self.clf.predict(x)
        ids = [cuisine.id for cuisine in cuisines]
        pred_cuisines = [self.dataset.id2cuisine[label] for label in ypred.astype(int)]
        self._write2csv(ids, pred_cuisines)

    def _save_model(self):
        print('Saving model...')
        pickle.dump(self.clf, open(os.path.join(BASE_DIR, self.model_name + '.model'), 'wb'))

    def load_model(self, model_path):
        print('Loading model...')
        self.clf = pickle.load(open(model_path, 'rb'))
