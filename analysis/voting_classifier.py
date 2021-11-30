from datetime import datetime
import itertools
from sklearn.model_selection import KFold
import pandas as pd
import json
from datetime import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

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
        tree = DecisionTreeClassifier()
        kneigh = KNeighborsClassifier(n_neighbors=101)

        vote = VotingClassifier(
                estimators=[('gnb', gnb), ('tree', tree), ('kneigh', kneigh)],
                voting='hard')

        # Set up kfold validation
        folds = KFold(n_splits=5, random_state=self.random_state, shuffle=True)

        for clf, label in zip([gnb, tree, kneigh, vote], ['Naive Bayes', 'MLP', 'Kneigh', 'Vote']):
            scores = cross_val_score(clf, self.df, self.target_names, scoring='accuracy', cv=5)
            print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

            # Add results to total for this feature combo
            self.results[str(label)] = {"accuracy" : scores.mean(), "stdev": scores.std()}

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

    with open("./data/results/vote.json", "w") as fl:
        json.dump(nb.to_json(), fl)
    
