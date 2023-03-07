import sklearn
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class RF:
    def __init__(self, data):
        self.training_set, self.testing_set = data
        self.train_data = self.training_set[0]
        self.train_label = self.training_set[1]
        self.test_data = self.testing_set[0]
        self.test_label = self.testing_set[1]
        self.model = RandomForestClassifier()
        self.predictions = []

    def train(self):
        self.model.fit(self.train_data, self.train_label)

    def test(self):
        self.predictions = self.model.predict(self.test_data)

    def loss_and_accuracy(self):
        self.predictions = np.array(self.predictions)
        self.test_label = np.array(self.test_label)
        accuracy = np.count_nonzero(self.test_label == self.predictions)/len(self.predictions)
        return accuracy

    '''
    evaluation the model with cross validation, AUC, confusion matrix. 
    '''
    def evaluation(self):
        pass
