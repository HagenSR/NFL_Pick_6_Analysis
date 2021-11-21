from sklearn import preprocessing
import pandas as pd


if __name__ == "__main__":
    res = pd.read_csv("data\encoded.csv")
    cols = list(res.columns)

    # Seperate targets from data
    target_names = res["score_bin"]

    res = res.drop("score_bin", 1)

    scaler = preprocessing.StandardScaler().fit(res)

    res = scaler.transform(res)
    res = pd.DataFrame(res, columns =cols[:-1])
    res["score_bin"] =target_names
    res.to_csv("data\encoded.csv")