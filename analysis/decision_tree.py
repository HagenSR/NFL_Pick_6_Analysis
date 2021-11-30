import itertools
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class DecisionTreeAnalysis:

    def __init__(self, file_path):
        # load encoded csv into numpy array
        self.df = pd.read_csv(file_path)
        # Seperate targets from data
        self.target_names = self.df["score_bin"]
        self.df = self.df.drop("score_bin", axis=1)

        # list of all feature indicies
        #self.features = [i for i in range(self.df.shape[1])]
                # list of all feature indicies
        self.features = []
        self.features.append(['schedule_week', 'spread_favorite', 'home_win', 'home_loss'])
        self.features.append(['spread_favorite', 'home_win', 'home_loss', 'away_win'])
        self.features.append(['schedule_week', 'team_home', 'spread_favorite', 'home_win', 'home_loss', 'away_win', 'away_loss'])
        self.features.append(['schedule_season','schedule_week','team_home','team_away','team_favorite_id','spread_favorite','over_under_line','stadium','home_win','home_loss','home_tie','away_win','away_loss','away_tie'])


        # Random state seed, results dict
        self.random_state = 101
        self.results = {}

    def train(self):
        clf = tree.DecisionTreeClassifier()
        # Set up kfold validation
        folds = KFold(n_splits=5, random_state=self.random_state, shuffle=True)

        # Iterate over all feature combinations
        incorrect = 0
        total = 0

        # run the NB model over each kfold
        flag = True
        for index in range(len(self.features)):
            combo = self.features[index]
            for train_index, test_index in folds.split(self.df):
                X_train, X_test = self.df.loc[train_index][combo], self.df.loc[test_index][combo]
                y_clf = clf.fit(X_train, self.target_names[train_index]).predict(X_test)
                if flag:
                    plt.figure(figsize=(25, 30))
                    tree.plot_tree(clf,filled=True, feature_names=combo, class_names=[str(i) for i in self.target_names])  
                    plt.savefig('./data/tree.png',format='png', bbox_inches='tight')
                    flag = False

                incorrect += (self.target_names[test_index] != y_clf ).sum()
                total += len(y_clf)

            # Add results to total for this feature combo
            self.results[str(combo)] = {"correct" : str(total - incorrect), "incorrect" : str(incorrect), 
                                    "total": str(total) , "accuracy" : (total- incorrect) / total }


    def to_json(self):
        # Sort results by accuracy and return
        return self.results

    def generate_matrix(self):
        gnb = tree.DecisionTreeClassifier()

        X_train, X_test, y_train, y_test = train_test_split(self.df, self.target_names, test_size=0.33, random_state=42)
        # run the NB model over each kfold
        y_gnb = gnb.fit(X_train, y_train).predict(X_test)

        tes = confusion_matrix(y_test, y_gnb)
        disp = ConfusionMatrixDisplay(tes, display_labels=["Home_Win", "Home_Loss", "Tie"])
        disp.plot()
        plt.title("Decision Tree")
        plt.show()


if __name__ == "__main__":
    dt = DecisionTreeAnalysis("data\encoded.csv")
    dt.generate_matrix()
    # dt.train()

    # with open("./data/results/decision_tree_results.json", "w") as fl:
    #     json.dump(dt.to_json(), fl)
    
