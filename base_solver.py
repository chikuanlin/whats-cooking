import csv

class BaseSolver:

    def __init__(self, dataset, in_features=6714):
        self.model_name = 'model_' + str(self.__class__.__name__).lower()
        self.target_file = 'submission_' +  str(self.__class__.__name__).lower() + '.csv'
        self.dataset = dataset
        self.in_features = in_features

    def train(self, x, y):
        # self._save_model()
        raise NotImplementedError

    def test(self, x, cuisines):
        # self._write2csv([], [])
        raise NotImplementedError
        
    def _save_model(self):
        raise NotImplementedError

    def load_model(self, model_path):
        raise NotImplementedError
    
    def _write2csv(self, ids, pred_cuisines):
        ''' Write results to csv file
        param ids          : list of id integers
        param pred_cuisines: list of cuisine strings 
        '''
        with open(self.target_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')
            csv_writer.writerow(['id', 'cuisine'])
            for i, cuisine in zip(ids, pred_cuisines):
                csv_writer.writerow([i, cuisine])

class ASolver(BaseSolver):
    def __init__(self, dataset):
        super(ASolver, self).__init__(dataset)

if __name__ == "__main__":
    solver = ASolver(123)
    solver._write2csv([1, 2, 3], ['a', 'b', 'c'])