from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import json

class NaiveBayesAnalysis:

    def __init__(self, file_path):
        # create data frame containing your data, each column can be accessed # by df['column   name']
        self.df = np.loadtxt(file_path, int, delimiter=",")

        self.X_train, self.X_test = train_test_split(self.df, test_size=0.33, random_state=42)

        self.target_names_train = self.X_train[:,-1]
        self.X_train = np.delete(self.X_train, -1, 1)

        self.target_names_test = self.X_test[:,-1]
        self.X_test = np.delete(self.X_test, -1, 1)

        # columns you want to model
        self.features = [i for i in range(self.X_train.shape[1])]
        #self.features = [4, 7, 8, 9, 10]
        self.y_gnb = None

    def train(self):
        # call Gaussian Naive Bayesian class with default parameters
        gnb = GaussianNB()

        # train model
        self.y_gnb = gnb.fit(self.X_train[:,self.features], self.target_names_train).predict(self.X_test[:,self.features])

    def to_json(self):
        incorrect = (self.target_names_test != self.y_gnb).sum()
        return {"correct" : str(self.X_test.shape[0] - incorrect), "incorrect" : str(incorrect), "total": str(self.X_test.shape[0]) }


if __name__ == "__main__":
    nb = NaiveBayesAnalysis("data\encoded.csv")
    nb.train()
    with open("./results.json", "w") as fl:
        json.dump(nb.to_json(), fl)
    
