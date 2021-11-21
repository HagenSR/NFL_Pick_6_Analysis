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
        self.df = pd.read_csv(file_path)

        # Seperate targets from data
        self.target_names = self.df["score_bin"]

        self.df = self.df.drop("score_bin", 1)

        # list of all feature indicies
        self.features = [self.df.columns[i] for i in range(len(self.df.columns))]

        # Random state seed, results dict
        self.random_state = 101
        self.results = {}

    def train(self):
        gnb = GaussianNB()

        # Set up kfold validation
        folds = KFold(n_splits=5, random_state=self.random_state, shuffle=True)

        # list all possible combinations of features
        combos = [x for l in range(20, len(self.features)) for x in itertools.combinations(self.features, l)]

        # Iterate over all feature combinations
        for index in range(len(combos)):
            combo = combos[index]
            if index % 1000 == 0:
                print("{0} out of {1}".format(index ,len(combos)))
            incorrect = 0
            total = 0

            # run the NB model over each kfold
            for train_index, test_index in folds.split(self.df):
                X_train, X_test = self.df.loc[train_index][list(combo)], self.df.loc[test_index][list(combo)]
                y_gnb = gnb.fit(X_train, self.target_names[train_index]).predict(X_test)

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

    with open("./data/results/naive_bayes.json", "w") as fl:
        json.dump(nb.to_json(), fl)
    
