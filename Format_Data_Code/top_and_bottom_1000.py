
import json

if __name__ == "__main__":
    jsn = []
    file_name = "data\\results\\naive_bayes"
    with open(file_name + ".json") as fl:
        jsn = json.load(fl)
    rtn = {}
    for key in list(jsn.keys())[0:1000]:
        rtn[key] = jsn[key]
    for key in list(jsn.keys())[-1000:]:
        rtn[key] = jsn[key]
    with open("{0}{1}.json".format(file_name, "_top_1000_and_bottom_1000"), "w") as fl:
        json.dump(rtn, fl)