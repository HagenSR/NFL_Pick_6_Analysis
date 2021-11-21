from sklearn import preprocessing
import pandas as pd


if __name__ == "__main__":
    res = pd.read_csv("data\encoded.csv")
    cols = list(res.columns)
    scaler = preprocessing.StandardScaler().fit(res)
    res = scaler.transform(res)
    res = pd.DataFrame(res, columns =cols)
    res.to_csv("data\encoded.csv")