from datetime import datetime
import itertools
from sklearn import svm
from sklearn.model_selection import KFold
import pandas as pd
import json
import matplotlib.pyplot as plt

class SupportVectorAnalysis:

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
        clf = svm.SVC()
        # Set up kfold validation
        folds = KFold(n_splits=5, random_state=self.random_state, shuffle=True)

         # list all possible combinations of features
        combos = [x for l in range(15, len(self.features)) for x in itertools.combinations(self.features, l)]
        #combos = [x for x in self.features]
        for index in range(len(combos)):
            print("Start: {0}, Index {1} out of {2}".format(datetime.now().time(), index, len(combos)))
            # Iterate over all feature combinations
            incorrect = 0
            total = 0
            combo = combos[index]
            # run the SVM model over each kfold
            for train_index, test_index in folds.split(self.df):
                X_train, X_test = self.df.loc[train_index][self.features], self.df.loc[test_index][self.features]
                y_clf = clf.fit(X_train, self.target_names[train_index]).predict(X_test)

                incorrect += (self.target_names[test_index] != y_clf ).sum()
                total += len(y_clf)

            # Add results to total for this feature combo
            self.results[str(combo)] = {"correct" : str(total - incorrect), "incorrect" : str(incorrect), 
                                    "total": str(total) , "accuracy" : (total- incorrect) / total }
            print("End: {0}".format(datetime.now().time()))


    def to_json(self):
        # Sort results by accuracy and return
        res = sorted(self.results.items(), key=lambda x:x[1]["accuracy"], reverse=True)
        return dict(res)


if __name__ == "__main__":
    dt = SupportVectorAnalysis("data\encoded.csv")
    dt.train()

    with open("./data/results/svm_results.json", "w") as fl:
        json.dump(dt.to_json(), fl)
    
