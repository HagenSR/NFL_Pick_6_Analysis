from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class NaiveBayesAnalysis:

    def __init__(self, file_path):
        # create data frame containing your data, each column can be accessed # by df['column   name']
        self.df = np.loadtxt(file_path, int, delimiter=",")

        # columns you want to model
        features = [4,7,8,9,10]

        X_train, X_test = train_test_split(self.df, test_size=0.33, random_state=42)

        target_names_train = X_train[:,-1]
        X_train = np.delete(X_train, 20, 1)

        target_names_test = X_test[:,-1]
        X_test = np.delete(X_test, 20, 1)

        # call Gaussian Naive Bayesian class with default parameters
        gnb = GaussianNB()

        # train model
        y_gnb = gnb.fit(X_train[:,features], target_names_train).predict(X_test[:,features])

        print("Correctly predicted {0} out of {1}".format(X_test.shape[0] - ((target_names_test != y_gnb).sum()), X_test.shape[0]))

if __name__ == "__main__":
    nb = NaiveBayesAnalysis("data\encoded.csv")
    
