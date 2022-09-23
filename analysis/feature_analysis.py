import json
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd


if __name__ == "__main__":

    acro_list = {}

    with open('data\\acronym_to_team.json') as fl:
        acro_list = json.load(fl)

    df = pd.read_csv("data\cleaned_data.csv")
    correct = 0
    total = 0
    for row in df.iterrows():
        row = row[1]
        if total == 0:
            total += 1
            continue
        if row["score_bin"] == "home_win":
            if row["team_favorite_id"] == acro_list[row["team_home"]]:
                correct+=1
        else:
            if row["team_favorite_id"] == acro_list[row["team_away"]]:
                correct+=1
        total+=1
        
    print(f"Correct: {correct} Total: {total} Percent: {correct/total}")


