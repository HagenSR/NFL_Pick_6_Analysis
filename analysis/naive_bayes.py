from datetime import datetime
import itertools
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
import pandas as pd
import json
from datetime import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class NaiveBayesAnalysis:

    def __init__(self, file_path):
        # load encoded csv into numpy array
        self.df = pd.read_csv(file_path)

        # Seperate targets from data
        self.target_names = self.df["score_bin"]

        self.df = self.df.drop("score_bin", axis=1)

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
        combos = [x for l in range(2, len(self.features)) for x in itertools.combinations(self.features, l)]

        # Iterate over all feature combinations
        for index in range(len(combos)):
            combo = combos[index]
            if index % 1000 == 0:
                print("Start: {0}, Index {1} out of {2}".format(datetime.now().time(), index, len(combos)))
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

    def generate_matrix(self):
        gnb = GaussianNB()

        features = ['team_home', 'home_win', 'home_loss', 'away_win', 'away_loss']

        X_train, X_test, y_train, y_test = train_test_split(self.df, self.target_names, test_size=0.33, random_state=42)
        # run the NB model over each kfold
        y_gnb = gnb.fit(X_train[features], y_train).predict(X_test[features])

        tes = confusion_matrix(y_test, y_gnb)
        disp = ConfusionMatrixDisplay(tes, display_labels=["Home_Win", "Home_Loss", "Tie"])
        disp.plot()
        plt.title("Naive Bayes")
        plt.show()
        


    def to_json(self):
        # Sort results by accuracy and return
        res = sorted(self.results.items(), key=lambda x:x[1]["accuracy"], reverse=True)
        return dict(res)


if __name__ == "__main__":
    nb = NaiveBayesAnalysis("data\encoded.csv")
    nb.train()
    #nb.generate_matrix()

    with open("./data/results/naive_bayes.json", "w") as fl:
        json.dump(nb.to_json(), fl)
    
