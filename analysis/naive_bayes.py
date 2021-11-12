import itertools
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import json

class NaiveBayesAnalysis:

    def __init__(self, file_path):
        # load encoded csv into numpy array
        self.df = np.loadtxt(file_path, int, delimiter=",")

        # Seperate targets from data
        self.target_names = self.df[:,-1]
        self.df = np.delete(self.df, -1, 1)

        # list of all feature indicies
        self.features = [i for i in range(self.df.shape[1])]

        # Random state seed, results dict
        self.random_state = 101
        self.results = {}

    def train(self):
        gnb = GaussianNB()

        # Set up kfold validation
        folds = KFold(n_splits=5, random_state=self.random_state, shuffle=True)

        # list all possible combinations of features
        combos = [x for l in range(2, len(self.features)) for x in itertools.combinations(self.features, l)]

        # Iterate over all feature combinations
        for index in range(len(combos)):
            combo = combos[index]
            if index % 1000 == 0:
                print("{0} out of {1}".format(index ,len(combos)))
            incorrect = 0
            total = 0

            # run the NB model over each kfold
            for train_index, test_index in folds.split(self.df):
                X_train, X_test = self.df[train_index], self.df[test_index]
                y_gnb = gnb.fit(X_train[:,combo], self.target_names[train_index]).predict(X_test[:,combo])

                incorrect += (self.target_names[test_index] != y_gnb ).sum()
                total += len(y_gnb)

            # Add results to total for this feature combo
            self.results[str(combo)] = {"correct" : str(total - incorrect), "incorrect" : str(incorrect), 
                                    "total": str(total) , "accuracy" : (total- incorrect) / total }


    def to_json(self):
        # Sort results by accuracy and return
        res = sorted(self.results.items(), key=lambda x:x[1]["accuracy"], reverse=True)
        return dict(res)


if __name__ == "__main__":
    nb = NaiveBayesAnalysis("data\encoded.csv")
    nb.train()

    with open("./results.json", "w") as fl:
        json.dump(nb.to_json(), fl)
    
