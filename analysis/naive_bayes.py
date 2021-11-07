from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class NaiveBayesAnalysis:

    def __init__(self, file_path):
        # create data frame containing your data, each column can be accessed # by df['column   name']
        self.df = np.loadtxt(file_path, int, delimiter=",")

        target_names = []
        for i in self.df:
            if i[-1] not in target_names:
                target_names.append(i[-1])

        # columns you want to model
        features = range(22)

        X_train, X_test = train_test_split(self.df, test_size=0.33, random_state=42)

        # call Gaussian Naive Bayesian class with default parameters
        gnb = GaussianNB()

        # train model
        y_gnb = gnb.fit(X_test[features], target_names).predict(X_train[features])

if __name__ == "__main__":
    nb = NaiveBayesAnalysis("data\csv.csv")
    
