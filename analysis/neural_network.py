from datetime import datetime
import itertools
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
import pandas as pd
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class MLPClassifierAnalysis:

    def __init__(self, file_path):
        # load encoded csv into numpy array
        self.df = pd.read_csv(file_path)
        # Seperate targets from data
        self.target_names = self.df["score_bin"]
        self.df = self.df.drop("score_bin", 1)

        self.features = []
        self.features.append(['schedule_week', 'spread_favorite', 'home_win', 'home_loss'])
        self.features.append(['spread_favorite', 'home_win', 'home_loss', 'away_win'])
        self.features.append(['schedule_week', 'team_home', 'spread_favorite', 'home_win', 'home_loss', 'away_win', 'away_loss'])
        self.features.append(['schedule_season','schedule_week','team_home','team_away','team_favorite_id','spread_favorite','over_under_line','stadium','home_win','home_loss','home_tie','away_win','away_loss','away_tie'])

        # Random state seed, results dict
        self.random_state = 101
        self.results = {}

    def train(self):
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=self.random_state, max_iter=10000)
        # Set up kfold validation
        folds = KFold(n_splits=5, random_state=self.random_state, shuffle=True)

        #combos = [x for x in self.features]
        for index in range(len(self.features)):
            print("Start: {0}, Index {1} out of {2}".format(datetime.now().time(), index, len(self.features)))
            # Iterate over all feature combinations
            incorrect = 0
            total = 0
            combo = self.features[index]
            # run the SVM model over each kfold
            for train_index, test_index in folds.split(self.df):
                X_train, X_test = self.df.loc[train_index][combo], self.df.loc[test_index][combo]
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

    def generate_matrix(self):
        gnb = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=self.random_state, max_iter=10000)

        features = ['schedule_week', 'team_home', 'spread_favorite', 'home_win', 'home_loss', 'away_win', 'away_loss']

        X_train, X_test, y_train, y_test = train_test_split(self.df, self.target_names, test_size=0.33, random_state=42)
        # run the NB model over each kfold
        y_gnb = gnb.fit(X_train[features], y_train).predict(X_test[features])

        tes = confusion_matrix(y_test, y_gnb)
        disp = ConfusionMatrixDisplay(tes, display_labels=["-22", "-15 | -21", "-10 | -14", "-7 | -9", "-5 | -6", "-3 | -4", "-1 | -2" ,"1 | 2", "3 | 4", "5 | 6", "7 | 9", "10 | 14", "15 | 21","22+"])
        disp.plot()
        plt.title("MLP Classifier")
        plt.show()


if __name__ == "__main__":
    dt = MLPClassifierAnalysis("data\encoded.csv")
    dt.generate_matrix()
    # dt.train()

    # with open("./data/results/MLPClassifier_Results.json", "w") as fl:
    #     json.dump(dt.to_json(), fl)
    
