import json

if __name__ == "__main__":
    headers = []
    with open("data\cleaned_data.csv") as fl:
        headers = fl.readline().replace("\n","").split(",")
    rtn = {}
    jsn = []
    with open("data\\naive_bayes_results.json") as fl:
        jsn = json.load(fl)
    for row in jsn:
        new_key = ""
        for index in row[0].replace("(", "").replace(")","").split(","):
            ind = int(index)
            new_key += headers[ind] + ", "
        rtn[new_key] = row[1]

    
    with open("data\\naive_bayes_results.json", "w") as fl:
        json.dump(rtn, fl)