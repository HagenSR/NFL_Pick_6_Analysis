import json

if __name__ == "__main__":
    csv = []
    with open("data\cleaned_data.csv") as fl:
        fl.readline()
        for line in fl:
            res = line.replace("\n", "").split(',')
            csv.append(res)
    rtn = {}
    total = 0
    for i in csv:
        if i[-1] not in rtn:
            rtn[i[-1]] = 1
        else:
            rtn[i[-1]] += 1
        total += 1
    rtn["total"] = total
    
    with open("data\\occurances.json", "w") as fl:
        json.dump(rtn, fl)
